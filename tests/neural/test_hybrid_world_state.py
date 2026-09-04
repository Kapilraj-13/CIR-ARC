"""Comprehensive tests for CIR-ARC Dual Neuro-Symbolic HybridSceneState and World-State v2.

Tests:
1. DenseLatentState & SymbolicSceneState data models
2. HybridSceneState dual representation and Cognitive Transformer token projection
3. ObjectAffordanceHead 9-class affordance prediction
4. TwoStagePointerHead ACTION6 pixel coordinate resolution
5. CategoricalEventEncoder discrete event detection
6. ActionConditionedTransitionModel latent dynamics
7. OnlineMechanicsTracker online physics belief update
8. PerceptionModel.to_hybrid_scene_state() end-to-end execution
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cir_arc.neural.world_state import (
    AFFORDANCE_NAMES,
    EVENT_TYPES,
    RELATION_TYPES,
    ActionEffect,
    DenseLatentState,
    GlobalStateData,
    HybridSceneState,
    MechanicsBelief,
    RelationGraph,
    SemanticEvent,
    SpatialRelation,
    StructuredObject,
    SymbolicSceneState,
    UncertaintySummary,
    WorldState,
)
from cir_arc.neural.perception.affordance_head import ObjectAffordanceHead
from cir_arc.neural.perception.pointer_head import TwoStagePointerHead
from cir_arc.neural.temporal.event_encoder import CategoricalEventEncoder, TemporalEventMemory
from cir_arc.neural.temporal.transition import ActionConditionedTransitionModel, OnlineMechanicsTracker
from cir_arc.neural.training.trainer import PerceptionModel


class TestDualNeuroSymbolicState:
    """Validates the Dual Hybrid Representation and safeguard."""

    def test_dense_latent_state(self) -> None:
        slots = torch.randn(24, 128)
        spatial = torch.randn(900, 128)
        pairwise = torch.randn(24, 24, 64)
        global_vec = torch.randn(128)

        dense = DenseLatentState(
            slot_embeddings=slots,
            spatial_features=spatial,
            pairwise_relational_latents=pairwise,
            global_scene_vector=global_vec,
        )

        assert dense.num_slots == 24
        assert dense.slot_dim == 128
        assert dense.pairwise_relational_latents.shape == (24, 24, 64)

    def test_symbolic_scene_state(self) -> None:
        obj = StructuredObject(
            slot_id=0,
            color=1,
            confidence=0.9,
            centroid=(0.5, 0.5),
            bbox=(0.4, 0.4, 0.6, 0.6),
            width=0.2,
            height=0.2,
            area=0.04,
            perimeter=0.8,
            aspect_ratio=1.0,
            shape_class=2,
            orientation=0,
            symmetries=(True, True, False, False),
            has_holes=False,
            extent_box=(12, 12, 18, 18),
            velocity=(0.0, 0.0),
            motion_direction="STILL",
            lifecycle_state="ACTIVE",
            affordances={"can_move": 0.8, "can_push": 0.2},
        )

        adj = np.zeros((1, 1, 14), dtype=np.float32)
        graph = RelationGraph(adj_matrix=adj, edge_list=[])

        sym = SymbolicSceneState(
            frame_index=1,
            grid_shape=(30, 30),
            objects=[obj],
            relations=[],
            scene_graph=graph,
            events=[SemanticEvent("MOVE", source_id=0)],
            global_state=GlobalStateData(score=10),
            mechanics_beliefs=MechanicsBelief(gravity=(1.0, 0.0), gravity_confidence=0.8),
        )

        assert sym.num_objects == 1
        assert sym.get_object_by_id(0) == obj
        assert len(sym.get_objects_by_color(1)) == 1
        assert sym.mechanics_beliefs.gravity == (1.0, 0.0)

    def test_hybrid_scene_state_and_cognitive_tokens(self) -> None:
        slots = torch.randn(24, 128)
        dense = DenseLatentState(slot_embeddings=slots)

        adj = np.zeros((2, 2, 14), dtype=np.float32)
        sym = SymbolicSceneState(
            frame_index=0,
            grid_shape=(10, 10),
            objects=[],
            relations=[],
            scene_graph=RelationGraph(adj_matrix=adj),
        )

        grid = np.zeros((10, 10), dtype=np.int8)
        hybrid = HybridSceneState(
            frame_index=0,
            grid_shape=(10, 10),
            raw_grid=grid,
            symbolic=sym,
            dense=dense,
        )

        # Backward compatibility alias
        assert isinstance(hybrid, WorldState)
        assert hybrid.num_objects == 0

        # Project to Cognitive Transformer tokens
        tokens = hybrid.to_cognitive_tokens(embed_dim=256)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (1, 25, 256)  # 1 global token + 24 slot tokens


class TestAffordanceAndPointerHeads:
    """Validates the affordance head and two-stage pointer resolution."""

    def test_object_affordance_head(self) -> None:
        head = ObjectAffordanceHead(slot_dim=128)
        slots = torch.randn(2, 24, 128)

        logits = head(slots, return_probs=False)
        assert logits.shape == (2, 24, 9)

        probs = head(slots, return_probs=True)
        assert probs.shape == (2, 24, 9)
        assert float(probs.detach().min()) >= 0.0 and float(probs.detach().max()) <= 1.0

        aff_dict = head.predict_affordance_dict(slots[0, 0])
        assert len(aff_dict) == 9
        for name in AFFORDANCE_NAMES:
            assert name in aff_dict
            assert 0.0 <= aff_dict[name] <= 1.0

    def test_two_stage_pointer_head(self) -> None:
        head = TwoStagePointerHead(slot_dim=128, feat_dim=128)
        slots = torch.randn(2, 24, 128)
        spatial = torch.randn(2, 100, 128)  # 10x10

        out = head(slots, spatial, H=10, W=10)
        assert out["slot_logits"].shape == (2, 24)
        assert out["selected_slot"].shape == (2,)
        assert out["pixel_heatmap"].shape == (2, 10, 10)
        assert out["coords_norm"].shape == (2, 2)
        assert out["coords_pixel"].shape == (2, 2)
        assert out["coords_xy"].shape == (2, 2)

        # Coordinates within bounds
        assert (out["coords_pixel"][:, 0] >= 0).all() and (out["coords_pixel"][:, 0] < 10).all()
        assert (out["coords_pixel"][:, 1] >= 0).all() and (out["coords_pixel"][:, 1] < 10).all()


class TestDynamicsAndEvents:
    """Validates event classification, temporal memory, and mechanics tracking."""

    def test_categorical_event_encoder(self) -> None:
        encoder = CategoricalEventEncoder(motion_threshold=0.05)

        obj1_t0 = StructuredObject(
            slot_id=1, color=2, confidence=1.0, centroid=(0.2, 0.2),
            bbox=(0.1, 0.1, 0.3, 0.3), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )
        obj2_t0 = StructuredObject(
            slot_id=2, color=3, confidence=1.0, centroid=(0.7, 0.7),
            bbox=(0.6, 0.6, 0.8, 0.8), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )

        # Frame t1: obj1 moved, obj2 vanished (DESTROY), new obj3 appeared (SPAWN)
        obj1_t1 = StructuredObject(
            slot_id=1, color=2, confidence=1.0, centroid=(0.4, 0.2),  # Moved
            bbox=(0.3, 0.1, 0.5, 0.3), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )
        obj3_t1 = StructuredObject(
            slot_id=3, color=4, confidence=1.0, centroid=(0.8, 0.8),  # Spawned
            bbox=(0.7, 0.7, 0.9, 0.9), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )

        events = encoder.encode_events([obj1_t0, obj2_t0], [obj1_t1, obj3_t1], step=1, action_id=2)
        event_types = [e.event_type for e in events]

        assert "MOVE" in event_types
        assert "DESTROY" in event_types
        assert "SPAWN" in event_types

    def test_temporal_event_memory(self) -> None:
        mem = TemporalEventMemory(max_history=10)
        events = [SemanticEvent("MOVE", source_id=1), SemanticEvent("COLLIDE", source_id=1, target_id=2)]
        mem.record_step(step=1, action_id=2, events=events)

        recent = mem.get_recent_events(5)
        assert len(recent) == 1
        assert recent[0]["step"] == 1
        assert len(recent[0]["events"]) == 2

        counts = mem.count_event_types()
        assert counts["MOVE"] == 1
        assert counts["COLLIDE"] == 1
        assert counts["TOGGLE"] == 0

    def test_action_conditioned_transition_model(self) -> None:
        model = ActionConditionedTransitionModel(slot_dim=128)
        slots = torch.randn(2, 24, 128)
        action = torch.tensor([1, 4])

        next_slots, delta_pos = model(slots, action)
        assert next_slots.shape == (2, 24, 128)
        assert delta_pos.shape == (2, 24, 2)

    def test_online_mechanics_tracker(self) -> None:
        tracker = OnlineMechanicsTracker()

        obj_a0 = StructuredObject(
            slot_id=1, color=2, confidence=1.0, centroid=(0.1, 0.5),
            bbox=(0.0, 0.4, 0.2, 0.6), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )
        obj_a1 = StructuredObject(
            slot_id=1, color=2, confidence=1.0, centroid=(0.3, 0.5),  # Fell down
            bbox=(0.2, 0.4, 0.4, 0.6), width=0.2, height=0.2, area=0.04,
            perimeter=0.8, aspect_ratio=1.0, shape_class=0, orientation=0,
            symmetries=(True, True, True, True), has_holes=False,
        )

        for _ in range(4):
            tracker.update_from_transition(action_id=0, prev_objects=[obj_a0], curr_objects=[obj_a1])

        belief = tracker.belief
        assert belief.gravity[0] == 1.0  # Inferred downward gravity direction
        assert belief.gravity_confidence > 0.5

        effects = tracker.compute_action_effects()
        assert len(effects) == 8
        assert effects[1].moves_player is True


class TestPerceptionModelEndToEnd:
    """Validates full end-to-end forward and to_hybrid_scene_state."""

    def test_perception_model_to_hybrid_scene_state(self) -> None:
        model = PerceptionModel(
            stem_hidden_dim=32,
            stem_out_dim=64,
            slot_dim=64,
            feat_dim=64,
            prop_hidden_dim=32,
            use_coordconv=True,
        )
        model.eval()

        grid = np.zeros((12, 12), dtype=np.int8)
        grid[2:5, 2:5] = 1
        grid[7:10, 7:10] = 2

        state = model.to_hybrid_scene_state(grid, obj_threshold=0.2)
        assert isinstance(state, HybridSceneState)
        assert state.grid_shape == (12, 12)
        assert state.dense.slot_embeddings.shape[0] == 24
        assert state.dense.pairwise_relational_latents.shape == (24, 24, 64)
        assert state.dense.spatial_features is not None

        # Verify summary output
        summary = state.summary()
        assert summary["has_dense_latents"] is True
        assert summary["dense_slot_count"] == 24
        assert summary["grid_shape"] == (12, 12)
