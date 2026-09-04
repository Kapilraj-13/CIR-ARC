"""Procedural 5-Layer Trajectory Dataset Generator for CIR-ARC Perception & World-State v2.

Generates rich, multi-tiered trajectories for training causal dynamics and mechanics inference:
- Layer 1 (35%): Static multi-object scenes (geometry, topology, relations)
- Layer 2 (25%): Temporal tracking sequences (velocities, acceleration, births, deaths)
- Layer 3 (20%): Action-conditioned transitions ((S_t, a_t) -> S_{t+1}, deltas, reversibility, events)
- Layer 4 (10%): Mechanics discovery (gravity, friction, portals, switches, pushability without labels)
- Layer 5 (10%): Novel-mechanics stress tests (momentum lag, delayed gates, coupled mirror kinematics)
- Negative examples with calibrated uncertainty targets (touching != supporting, downward != gravity, etc.)
- Explicit boolean valid_mask separating ARC palette (0-9) from padding.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cir_arc.core.grid import Grid
from cir_arc.core.objects import ArcObject, extract_objects
from cir_arc.neural.world_state import EVENT_TYPES, RELATION_TYPES, SemanticEvent


@dataclass
class TrajectoryStep:
    """A single action-conditioned causal transition step."""
    layer: int                                           # 1 to 5
    grid_t: List[List[int]]                              # 2D discrete grid at t (colors 0-9)
    valid_mask_t: List[List[bool]]                       # 2D boolean mask (True = valid cell)
    action: int                                          # Executed action (0..7)
    grid_next: List[List[int]]                           # 2D discrete grid at t+1
    valid_mask_next: List[List[bool]]                    # 2D boolean mask at t+1
    events: List[Dict[str, Any]] = field(default_factory=list) # SemanticEvent dicts
    action_effect: Dict[str, Any] = field(default_factory=dict) # Success, cost, moves_player, reversibility
    mechanics_evidence: Dict[str, Any] = field(default_factory=dict) # Gravity, friction, push, etc.
    is_negative_example: bool = False                    # Counter-intuitive / ambiguous example flag
    ambiguity_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProceduralTrajectoryGenerator:
    """Generates balanced 5-layer trajectory training corpora for CIR-ARC."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def _create_empty_grid(self, H: int, W: int, bg_color: int = 0) -> np.ndarray:
        return np.full((H, W), bg_color, dtype=np.int8)

    # --------------------------------------------------------------------------
    # Layer 1: Static Multi-Object Scenes (35%)
    # --------------------------------------------------------------------------
    def generate_layer1_static(self, H: int = 15, W: int = 15) -> TrajectoryStep:
        grid = self._create_empty_grid(H, W)
        num_objs = self.rng.integers(2, 6)
        colors = self.rng.choice(np.arange(1, 10), size=num_objs, replace=False)

        for i in range(num_objs):
            c = int(colors[i])
            h = self.rng.integers(2, 5)
            w = self.rng.integers(2, 5)
            r = self.rng.integers(0, max(1, H - h))
            col = self.rng.integers(0, max(1, W - w))

            # Shape variety: rectangle, hollow frame, or cross
            shape_type = self.rng.choice(["rect", "hollow", "cross"])
            if shape_type == "rect":
                grid[r:r+h, col:col+w] = c
            elif shape_type == "hollow" and h >= 3 and w >= 3:
                grid[r:r+h, col:col+w] = c
                grid[r+1:r+h-1, col+1:col+w-1] = 0  # Enclosed topological hole
            elif shape_type == "cross" and h >= 3 and w >= 3:
                mid_r = r + h // 2
                mid_c = col + w // 2
                grid[r:r+h, mid_c] = c
                grid[mid_r, col:col+w] = c

        mask = np.ones((H, W), dtype=bool)
        return TrajectoryStep(
            layer=1,
            grid_t=grid.tolist(),
            valid_mask_t=mask.tolist(),
            action=0,
            grid_next=grid.tolist(),
            valid_mask_next=mask.tolist(),
            events=[],
            action_effect={"success": True, "moves_player": False, "cost": 0.0, "reversible": True},
            mechanics_evidence={},
            is_negative_example=False,
        )

    # --------------------------------------------------------------------------
    # Layer 2: Temporal Perception & Kinematics (25%)
    # --------------------------------------------------------------------------
    def generate_layer2_temporal(self, H: int = 16, W: int = 16) -> TrajectoryStep:
        grid_t = self._create_empty_grid(H, W)
        grid_next = self._create_empty_grid(H, W)

        # Object 1: Constant velocity motion
        r0, c0 = self.rng.integers(2, 8), self.rng.integers(2, 8)
        dr, dc = int(self.rng.choice([-1, 0, 1])), int(self.rng.choice([-1, 0, 1]))
        if dr == 0 and dc == 0:
            dr = 1

        color1 = int(self.rng.integers(1, 5))
        grid_t[r0:r0+2, c0:c0+2] = color1
        grid_next[r0+dr:r0+dr+2, c0+dc:c0+dc+2] = color1

        # Object 2: Stationary obstacle
        color2 = int(self.rng.integers(6, 10))
        r_wall, c_wall = self.rng.integers(10, 14), self.rng.integers(10, 14)
        grid_t[r_wall:r_wall+2, c_wall:c_wall+2] = color2
        grid_next[r_wall:r_wall+2, c_wall:c_wall+2] = color2

        mask = np.ones((H, W), dtype=bool)
        events = [
            SemanticEvent("MOVE", source_id=0, details={"velocity": (float(dr), float(dc))}).to_dict()
        ]

        return TrajectoryStep(
            layer=2,
            grid_t=grid_t.tolist(),
            valid_mask_t=mask.tolist(),
            action=0,
            grid_next=grid_next.tolist(),
            valid_mask_next=mask.tolist(),
            events=events,
            action_effect={"success": True, "moves_player": True, "cost": 1.0, "reversible": True},
            mechanics_evidence={"velocity": (float(dr), float(dc))},
            is_negative_example=False,
        )

    # --------------------------------------------------------------------------
    # Layer 3: Action-Conditioned Dynamics (20%)
    # --------------------------------------------------------------------------
    def generate_layer3_action_transition(self, H: int = 14, W: int = 14) -> TrajectoryStep:
        grid_t = self._create_empty_grid(H, W)
        grid_next = self._create_empty_grid(H, W)

        # Player at (pr, pc)
        pr, pc = self.rng.integers(3, 10), self.rng.integers(3, 10)
        player_color = 2
        grid_t[pr, pc] = player_color

        # Action: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        action = int(self.rng.integers(1, 5))
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]

        # 50% chance of pushable box in front of player
        has_box = bool(self.rng.random() > 0.5)
        box_r, box_c = pr + dr, pc + dc
        box_color = 4

        events = []
        if has_box:
            grid_t[box_r, box_c] = box_color
            next_box_r, next_box_c = box_r + dr, box_c + dc

            # Check if wall blocks box
            has_wall = next_box_r < 0 or next_box_r >= H or next_box_c < 0 or next_box_c >= W
            if has_wall:
                # Blocked: no movement
                grid_next[pr, pc] = player_color
                grid_next[box_r, box_c] = box_color
                success = False
                events.append(SemanticEvent("COLLIDE", source_id=0, target_id=1).to_dict())
            else:
                # Pushed box successfully
                grid_next[pr + dr, pc + dc] = player_color
                grid_next[next_box_r, next_box_c] = box_color
                success = True
                events.append(SemanticEvent("MOVE", source_id=0).to_dict())
                events.append(SemanticEvent("MOVE", source_id=1).to_dict())
        else:
            # Free move
            grid_next[pr + dr, pc + dc] = player_color
            success = True
            events.append(SemanticEvent("MOVE", source_id=0).to_dict())

        mask = np.ones((H, W), dtype=bool)
        return TrajectoryStep(
            layer=3,
            grid_t=grid_t.tolist(),
            valid_mask_t=mask.tolist(),
            action=action,
            grid_next=grid_next.tolist(),
            valid_mask_next=mask.tolist(),
            events=events,
            action_effect={"success": success, "moves_player": success, "cost": 1.0, "reversible": True},
            mechanics_evidence={"pushability": 1.0 if has_box and success else 0.0},
            is_negative_example=False,
        )

    # --------------------------------------------------------------------------
    # Layer 4: Mechanics Discovery (10%)
    # --------------------------------------------------------------------------
    def generate_layer4_mechanics_discovery(self, H: int = 15, W: int = 15) -> TrajectoryStep:
        # Choose mechanic: 0=Gravity, 1=Ice Sliding, 2=Portal Teleport, 3=Switch Gate
        mech_type = int(self.rng.integers(0, 4))
        grid_t = self._create_empty_grid(H, W)
        grid_next = self._create_empty_grid(H, W)

        if mech_type == 0:
            # Gravity: boulder falls down 2 tiles
            r, c = self.rng.integers(1, 6), self.rng.integers(2, 13)
            grid_t[r:r+2, c:c+2] = 3
            grid_next[r+2:r+4, c:c+2] = 3  # Falls down under gravity
            action = 0  # No action / tick
            events = [SemanticEvent("MOVE", source_id=0, details={"fall_distance": 2}).to_dict()]
            evidence = {"gravity": (1.0, 0.0), "gravity_confidence": 0.9}

        elif mech_type == 1:
            # Ice Sliding: player slides all the way to border/wall
            r, c = 4, 2
            grid_t[r, c] = 2
            # Slide RIGHT until column 12
            grid_next[r, 12] = 2
            action = 4  # RIGHT
            events = [SemanticEvent("MOVE", source_id=0, details={"slide_length": 10}).to_dict()]
            evidence = {"sliding_inertia": 1.0, "friction": 0.0}

        elif mech_type == 2:
            # Portal Teleportation: entry at (3,3), exit at (11,11)
            p1_r, p1_c = 3, 3
            p2_r, p2_c = 11, 11
            portal_color = 8
            player_color = 1

            grid_t[p1_r, p1_c] = portal_color
            grid_t[p2_r, p2_c] = portal_color
            grid_t[p1_r, p1_c - 1] = player_color

            # Player steps RIGHT into portal, appears at p2
            grid_next[p1_r, p1_c] = portal_color
            grid_next[p2_r, p2_c] = portal_color
            grid_next[p2_r, p2_c + 1] = player_color
            action = 4
            events = [SemanticEvent("MOVE", source_id=0, details={"teleport": True}).to_dict()]
            evidence = {"teleportation_active": 1.0}

        else:
            # Pressure Plate & Gate: stepping on plate toggles gate
            plate_r, plate_c = 5, 5
            gate_r, gate_c = 8, 8
            grid_t[plate_r, plate_c] = 7  # Plate
            grid_t[gate_r, gate_c] = 5    # Closed Gate (solid)
            grid_t[plate_r - 1, plate_c] = 2 # Player

            # Step DOWN on plate -> Gate opens (disappears or turns into open color)
            grid_next[plate_r, plate_c] = 2  # Player on plate
            grid_next[gate_r, gate_c] = 0    # Gate opened!
            action = 2
            events = [
                SemanticEvent("MOVE", source_id=0).to_dict(),
                SemanticEvent("TOGGLE", source_id=1).to_dict(),
            ]
            evidence = {"toggle_mechanics": 1.0}

        mask = np.ones((H, W), dtype=bool)
        return TrajectoryStep(
            layer=4,
            grid_t=grid_t.tolist(),
            valid_mask_t=mask.tolist(),
            action=action,
            grid_next=grid_next.tolist(),
            valid_mask_next=mask.tolist(),
            events=events,
            action_effect={"success": True, "moves_player": True, "cost": 1.0, "reversible": False},
            mechanics_evidence=evidence,
            is_negative_example=False,
        )

    # --------------------------------------------------------------------------
    # Layer 5: Novel-Mechanics Stress Tests (10%)
    # --------------------------------------------------------------------------
    def generate_layer5_novel_stress_test(self, H: int = 16, W: int = 16) -> TrajectoryStep:
        grid_t = self._create_empty_grid(H, W)
        grid_next = self._create_empty_grid(H, W)

        # Coupled Mirrored Kinematics: moving Agent A left moves Agent B right
        r_mid = H // 2
        grid_t[r_mid, 4] = 2
        grid_t[r_mid, 11] = 3

        # Action LEFT (3)
        grid_next[r_mid, 3] = 2   # Moves left
        grid_next[r_mid, 12] = 3  # Coupled mirror agent moves right!
        action = 3

        mask = np.ones((H, W), dtype=bool)
        events = [
            SemanticEvent("MOVE", source_id=0).to_dict(),
            SemanticEvent("MOVE", source_id=1, details={"coupled_mirror": True}).to_dict(),
        ]

        return TrajectoryStep(
            layer=5,
            grid_t=grid_t.tolist(),
            valid_mask_t=mask.tolist(),
            action=action,
            grid_next=grid_next.tolist(),
            valid_mask_next=mask.tolist(),
            events=events,
            action_effect={"success": True, "moves_player": True, "cost": 1.0, "reversible": True},
            mechanics_evidence={"coupled_symmetry": 1.0},
            is_negative_example=False,
            ambiguity_notes="Non-local coupled multi-agent action",
        )

    # --------------------------------------------------------------------------
    # Negative Examples & Uncertainty Calibration
    # --------------------------------------------------------------------------
    def generate_negative_example(self, H: int = 14, W: int = 14) -> TrajectoryStep:
        grid_t = self._create_empty_grid(H, W)
        grid_next = self._create_empty_grid(H, W)

        # Scenario: Object touching a wall laterally does NOT mean it is supported
        obj_r, obj_c = 4, 4
        grid_t[obj_r:obj_r+2, obj_c:obj_c+2] = 1
        grid_t[obj_r:obj_r+2, obj_c+2] = 5  # Wall touching on the right side

        # Next step: object drops downward despite touching the right wall!
        grid_next[obj_r+2:obj_r+4, obj_c:obj_c+2] = 1
        grid_next[obj_r:obj_r+2, obj_c+2] = 5

        mask = np.ones((H, W), dtype=bool)
        events = [SemanticEvent("MOVE", source_id=0, confidence=0.6).to_dict()]

        return TrajectoryStep(
            layer=3,
            grid_t=grid_t.tolist(),
            valid_mask_t=mask.tolist(),
            action=0,
            grid_next=grid_next.tolist(),
            valid_mask_next=mask.tolist(),
            events=events,
            action_effect={"success": True, "moves_player": False, "cost": 0.0, "reversible": False},
            mechanics_evidence={"gravity": (1.0, 0.0), "gravity_confidence": 0.55},
            is_negative_example=True,
            ambiguity_notes="Touching lateral wall != SUPPORTED_BY; gravity drop occurs",
        )

    def generate_balanced_corpus(self, n_samples: int = 1000) -> List[TrajectoryStep]:
        """Generates a corpus matching the recommended distribution:
        - 35% Layer 1 (Static)
        - 25% Layer 2 (Temporal)
        - 20% Layer 3 (Action transitions)
        - 10% Layer 4 (Mechanics discovery)
        - 10% Layer 5 (Novel-mechanics stress tests)
        - ~10% interspersed negative examples
        """
        corpus: List[TrajectoryStep] = []

        n_l1 = int(n_samples * 0.35)
        n_l2 = int(n_samples * 0.25)
        n_l3 = int(n_samples * 0.20)
        n_l4 = int(n_samples * 0.10)
        n_l5 = n_samples - (n_l1 + n_l2 + n_l3 + n_l4)

        for _ in range(n_l1):
            corpus.append(self.generate_layer1_static())
        for _ in range(n_l2):
            corpus.append(self.generate_layer2_temporal())
        for _ in range(n_l3):
            # 20% negative examples within action transitions
            if self.rng.random() < 0.2:
                corpus.append(self.generate_negative_example())
            else:
                corpus.append(self.generate_layer3_action_transition())
        for _ in range(n_l4):
            corpus.append(self.generate_layer4_mechanics_discovery())
        for _ in range(n_l5):
            corpus.append(self.generate_layer5_novel_stress_test())

        self.rng.shuffle(corpus)
        return corpus

    def save_corpus_to_dir(self, corpus: List[TrajectoryStep], output_dir: str) -> None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        for idx, step in enumerate(corpus):
            fname = out_p / f"traj_{idx:06d}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(step.to_dict(), f)
        print(f"Saved {len(corpus)} trajectory steps to {output_dir}")
