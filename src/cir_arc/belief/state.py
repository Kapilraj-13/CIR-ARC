"""Unified Epistemic Belief State for ARC-AGI-3 Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from cir_arc.belief.facts import Fact, FactSet, FactType, Provenance
from cir_arc.belief.uncertainty import UncertaintyModel
from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData, GameState


@dataclass
class ObservedObjectState:
    """Tracked state of an individual object in the belief state."""
    object_id: str
    color: int
    pixels: List[Tuple[int, int]]
    centroid: Tuple[float, float]
    size: int
    role: str = "unknown"                       # "player", "wall", "key", "door", "goal", "collectible", "unknown"
    is_interactive: bool = False
    is_passable: Optional[bool] = None
    provenance: Provenance = Provenance.OBSERVED
    last_seen_step: int = 0


@dataclass
class BeliefState:
    """Epistemic belief state separating direct observations, inferred facts, and hypotheses."""
    game_id: str
    step_count: int = 0
    current_game_state: GameState = GameState.NOT_PLAYED
    grid_shape: Tuple[int, int] = (10, 10)
    player_location: Optional[Tuple[int, int]] = None
    player_color: int = 9

    # Epistemic Containers
    facts: FactSet = field(default_factory=FactSet)
    uncertainty: UncertaintyModel = field(default_factory=UncertaintyModel)
    observed_objects: Dict[str, ObservedObjectState] = field(default_factory=dict)
    visited_states: Set[str] = field(default_factory=set)
    visited_coordinates: Set[Tuple[int, int]] = field(default_factory=set)

    # Transition and Hypothesis State
    known_transitions: Dict[str, Any] = field(default_factory=dict)
    uncertain_transitions: List[Any] = field(default_factory=list)
    failed_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Goals and Planning
    goal_hypotheses: List[Any] = field(default_factory=list)
    active_goal: Optional[Any] = None
    current_plan: List[Action] = field(default_factory=list)
    confidence: float = 0.5
    action_history: List[Tuple[Action, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.uncertainty.resize(self.grid_shape[0], self.grid_shape[1])
        # Register basic facts
        self.facts.add_or_update(
            FactType.OBJECT_ROLE,
            subject=f"color_{self.player_color}",
            predicate="is_player",
            value=True,
            provenance=Provenance.FACT,
            confidence=1.0,
        )

    def update_from_frame(self, frame: FrameData) -> None:
        """Update the belief state with a new incoming observation frame."""
        self.step_count = frame.step_count
        self.current_game_state = frame.state
        comp = frame.grid.composite()
        H, W = comp.shape
        self.grid_shape = (H, W)
        self.uncertainty.resize(H, W)

        # Record visited grid hash
        frame_hash = frame.hash()
        self.visited_states.add(frame_hash)

        # 1. Update Player Location
        player_locs = [tuple(p) for p in np.argwhere(comp == self.player_color)]
        if player_locs:
            r, c = int(player_locs[0][0]), int(player_locs[0][1])
            self.player_location = (r, c)
            self.visited_coordinates.add((r, c))
            self.uncertainty.record_visit(r, c)
            self.facts.add_or_update(
                FactType.OBJECT_POSITION,
                subject="player",
                predicate="position",
                value=(r, c),
                provenance=Provenance.OBSERVED,
                confidence=1.0,
            )

        # 2. Extract and track distinct color clusters
        unique_colors = np.unique(comp)
        for color_val in unique_colors:
            color = int(color_val)
            if color == 0:
                self.facts.add_or_update(
                    FactType.PASSABILITY,
                    subject="color_0",
                    predicate="is_passable",
                    value=True,
                    provenance=Provenance.FACT,
                    confidence=1.0,
                )
                continue

            pixels = [tuple(p) for p in np.argwhere(comp == color)]
            if not pixels:
                continue

            obj_id = f"obj_color_{color}"
            centroid = (
                float(sum(p[0] for p in pixels) / len(pixels)),
                float(sum(p[1] for p in pixels) / len(pixels)),
            )

            # Determine probable role based on color and properties
            role = "unknown"
            if color == self.player_color:
                role = "player"
            elif color in (14, 3, 2) and len(pixels) <= 4:
                role = "goal_candidate"
            elif color in (11, 6) and len(pixels) <= 4:
                role = "key_candidate"
            elif color in (8, 7) and len(pixels) >= 1:
                role = "door_candidate"
            elif color in (5, 1) and len(pixels) > 5:
                role = "wall_candidate"

            self.observed_objects[obj_id] = ObservedObjectState(
                object_id=obj_id,
                color=color,
                pixels=pixels,
                centroid=centroid,
                size=len(pixels),
                role=role,
                last_seen_step=self.step_count,
            )

            self.facts.add_or_update(
                FactType.OBJECT_EXISTS,
                subject=obj_id,
                predicate="color",
                value=color,
                provenance=Provenance.OBSERVED,
                confidence=1.0,
            )

    def record_transition(self, action: Action, result_state_hash: str, was_passable: bool, target_pos: Tuple[int, int], target_color: int) -> None:
        """Record the physical result of taking an action against an environment cell."""
        self.action_history.append((action, result_state_hash))

        # Update passability fact
        self.facts.add_or_update(
            FactType.PASSABILITY,
            subject=f"color_{target_color}",
            predicate="is_passable",
            value=was_passable,
            provenance=Provenance.OBSERVED,
            confidence=1.0 if was_passable else 0.9,
        )

        # Update uncertainty
        self.uncertainty.update_color_certainty(target_color, 1.0)
        self.uncertainty.update_interaction_certainty(target_color, int(action.action_type), 1.0)

    def get_passable_mask(self, comp_grid: np.ndarray) -> np.ndarray:
        """Compute boolean 2D mask (H, W) where True indicates passable cell."""
        H, W = comp_grid.shape
        mask = np.ones((H, W), dtype=bool)

        blocked_colors = self.facts.get_known_blocked_colors()
        for r in range(H):
            for c in range(W):
                color = int(comp_grid[r, c])
                if color == 0:
                    mask[r, c] = True
                elif color in blocked_colors:
                    mask[r, c] = False
                elif color == self.player_color:
                    mask[r, c] = True
                elif color == 5:  # Default solid wall
                    mask[r, c] = False
                else:
                    # If unknown, tentatively allow with slight cost or check role
                    fact = self.facts.get_fact(FactType.PASSABILITY, f"color_{color}", "is_passable")
                    if fact is not None and fact.value is False:
                        mask[r, c] = False
                    else:
                        mask[r, c] = True
        return mask

    def summary(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "player_location": self.player_location,
            "objects_count": len(self.observed_objects),
            "facts_count": len(self.facts),
            "visited_coords_count": len(self.visited_coordinates),
            "confidence": self.confidence,
            "epistemic_entropy": self.uncertainty.compute_total_epistemic_entropy(),
        }
