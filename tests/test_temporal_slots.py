import numpy as np
import pytest
import torch

from cir_arc.neural.perception.embedding import ColorEmbedding
from cir_arc.neural.perception.slot_attention import SlotAttention
from cir_arc.neural.temporal.kinematics import classify_motion_direction, compute_velocity
from cir_arc.neural.temporal.tracker import TemporalSlotTracker, TrackedSlot


class TestColorEmbedding16:
    def test_17_token_embedding(self):
        embed = ColorEmbedding(num_colors=17, embed_dim=48)
        grid = torch.randint(0, 17, (2, 8, 8), dtype=torch.long)
        out = embed(grid)
        assert out.shape == (2, 8, 8, 48)


class TestKinematics:
    def test_velocity_computation(self):
        p1 = (1.0, 1.0)
        p2 = (3.0, 5.0)
        v = compute_velocity(p1, p2, dt=2.0)
        assert v == (1.0, 2.0)

    def test_direction_classification(self):
        assert classify_motion_direction(0.0, 0.0) == "STILL"
        assert classify_motion_direction(-1.0, 0.0) == "N"
        assert classify_motion_direction(1.0, 0.0) == "S"
        assert classify_motion_direction(0.0, -1.0) == "W"
        assert classify_motion_direction(0.0, 1.0) == "E"
        assert classify_motion_direction(-1.0, 1.0) == "NE"
        assert classify_motion_direction(1.0, 1.0) == "SE"
        assert classify_motion_direction(-1.0, -1.0) == "NW"
        assert classify_motion_direction(1.0, -1.0) == "SW"


class TestTemporalSlotTracker:
    def test_tracker_spawns_and_tracks_moving_object(self):
        tracker = TemporalSlotTracker(objectness_threshold=0.3, slot_dim=64)

        # Frame 1: Single object at (10, 10)
        K = 4
        slots_f1 = torch.randn(K, 64)
        obj_f1 = torch.tensor([0.9, 0.1, 0.1, 0.1])
        pos_f1 = torch.tensor([[10.0 / 30.0, 10.0 / 30.0], [0, 0], [0, 0], [0, 0]])
        color_f1 = torch.zeros(K, 16)
        color_f1[0, 9] = 1.0  # Blue

        tracks_f1 = tracker.update_from_perception(
            slots=slots_f1,
            objectness=obj_f1,
            pos_preds=pos_f1,
            color_preds=color_f1,
            H=30,
            W=30,
        )
        assert len(tracks_f1) == 1
        t_id = tracks_f1[0].track_id
        assert tracks_f1[0].predicted_color == 9
        assert pytest.approx(tracks_f1[0].centroid[0], 0.1) == 10.0
        assert tracks_f1[0].lifecycle_state == "SPAWNED"

        # Frame 2: Object moves to (11, 10) (Down)
        slots_f2 = slots_f1.clone() + 0.05 * torch.randn_like(slots_f1)
        obj_f2 = torch.tensor([0.95, 0.1, 0.1, 0.1])
        pos_f2 = torch.tensor([[11.0 / 30.0, 10.0 / 30.0], [0, 0], [0, 0], [0, 0]])
        color_f2 = color_f1.clone()

        tracks_f2 = tracker.update_from_perception(
            slots=slots_f2,
            objectness=obj_f2,
            pos_preds=pos_f2,
            color_preds=color_f2,
            H=30,
            W=30,
        )
        assert len(tracks_f2) == 1
        assert tracks_f2[0].track_id == t_id  # Invariant: Persistent identity!
        assert tracks_f2[0].lifecycle_state == "ACTIVE"
        assert pytest.approx(tracks_f2[0].centroid[0], 0.1) == 11.0
        assert tracks_f2[0].motion_direction == "S"  # Downwards is South in grid

    def test_slot_persistence_invariant(self):
        """Acceptance Criteria check: >= 90% slot persistence accuracy on moving tracks."""
        tracker = TemporalSlotTracker(objectness_threshold=0.3, slot_dim=64)
        K = 6
        N_steps = 20
        persisted_count = 0
        total_eval_steps = N_steps - 1

        # Generate base slots for 3 persistent objects
        base_slots = torch.randn(3, 64)
        positions = [(5.0, 5.0), (10.0, 15.0), (20.0, 20.0)]
        colors = [9, 11, 14]

        # Step 0
        slots_0 = torch.zeros(K, 64)
        slots_0[:3] = base_slots
        obj_0 = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        pos_0 = torch.zeros(K, 2)
        color_0 = torch.zeros(K, 16)
        for i in range(3):
            pos_0[i] = torch.tensor([positions[i][0] / 30.0, positions[i][1] / 30.0])
            color_0[i, colors[i]] = 1.0

        initial_tracks = tracker.update_from_perception(slots_0, obj_0, pos_preds=pos_0, color_preds=color_0, H=30, W=30)
        assert len(initial_tracks) == 3
        expected_ids = {t.track_id for t in initial_tracks}

        # Step 1..N
        for step in range(1, N_steps):
            slots_t = torch.zeros(K, 64)
            pos_t = torch.zeros(K, 2)
            color_t = torch.zeros(K, 16)
            obj_t = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])

            # Perturb embeddings slightly & move positions
            for i in range(3):
                slots_t[i] = base_slots[i] + 0.05 * torch.randn(64)
                new_r = min(28.0, positions[i][0] + 0.5 * step)
                new_c = min(28.0, positions[i][1] + 0.5 * step)
                pos_t[i] = torch.tensor([new_r / 30.0, new_c / 30.0])
                color_t[i, colors[i]] = 1.0

            updated_tracks = tracker.update_from_perception(slots_t, obj_t, pos_preds=pos_t, color_preds=color_t, H=30, W=30)
            active_ids = {t.track_id for t in updated_tracks if t.lifecycle_state == "ACTIVE"}
            if active_ids == expected_ids:
                persisted_count += 1

        persistence_rate = persisted_count / total_eval_steps
        assert persistence_rate >= 0.90, f"Persistence rate {persistence_rate:.2f} fell below 0.90 requirement!"
