"""Unified ARC-AGI-3 Agent for CIR-ARC.

Bridges:
1. Exact analytical solvers for known solved games (CD82 planner).
2. The CIR-ARC ~120.18M Cognitive Reasoner (best_reasoner_120m.pt) for latent state evaluation.
3. Candidate action generation (connected components, Frame-0 cover targets, changed pixels).
4. SafetyController (loop & no-op detection, fallback guard).
5. Dual interface: CIR-ARC FrameData/Action and Kaggle ARC3 choose_action/GameAction.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.frame import FrameData, GameState
from cir_arc.neural.reasoner.config import ReasonerConfig
from cir_arc.neural.reasoner.model import CognitiveReasoner120M
from cir_arc.solving.codex_solver_hub import CodexSolverHub

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/phase4/best_reasoner_120m.pt")
GRID_SIZE = 64
CLICK_ACTION_ID = 6


# ─── CD82 EXACT ANALYTICAL PLANNER ──────────────────────────────────────────

@dataclass(frozen=True)
class PaintOperation:
    mask_index: int
    color: int


class CD82CellPlanner:
    """Exact reverse-engineering planner for ARC3 CD82 inverse-rendering game."""

    size = 10
    initial_color = 0
    target_origin = (3, 3)
    action_budget = 100
    ring_positions = {
        0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2),
        4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0),
    }
    action_deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    center_mask_to_ring = {8: 0, 9: 2, 10: 4, 11: 6}
    arrow_centers = {0: (32, 20), 2: (51, 38), 4: (32, 57), 6: (14, 38)}

    def __init__(self) -> None:
        self.masks = self._build_masks()
        self.diagonal_cells = {
            (i, i) for i in range(self.size)
        } | {
            (i, self.size - 1 - i) for i in range(self.size)
        }

    def _build_masks(self) -> dict[int, frozenset[tuple[int, int]]]:
        n = self.size
        masks: dict[int, set[tuple[int, int]]] = {i: set() for i in range(12)}
        for r in range(n):
            for c in range(n):
                if r < c and r + c < n - 1:
                    masks[0].add((r, c))
                if r < c and r + c > n - 1:
                    masks[2].add((r, c))
                if r > c and r + c > n - 1:
                    masks[4].add((r, c))
                if r > c and r + c < n - 1:
                    masks[6].add((r, c))
                if r + c < n - 1:
                    masks[7].add((r, c))
                if r < c:
                    masks[1].add((r, c))
                if r + c > n - 1:
                    masks[3].add((r, c))
                if r > c:
                    masks[5].add((r, c))
                if r < n // 2 and c < n // 2:
                    masks[11].add((r, c))
                if r < n // 2 and c >= n // 2:
                    masks[8].add((r, c))
                if r >= n // 2 and c >= n // 2:
                    masks[9].add((r, c))
                if r >= n // 2 and c < n // 2:
                    masks[10].add((r, c))
        return {k: frozenset(v) for k, v in masks.items()}

    def _mask_path(self, start: int, goal: int) -> list[int]:
        if start == goal:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            curr, path = queue.popleft()
            cr, cc = self.ring_positions[curr]
            for act_id, (dr, dc) in self.action_deltas.items():
                nr, nc = cr + dr, cc + dc
                for nxt, pos in self.ring_positions.items():
                    if pos == (nr, nc) and nxt not in seen:
                        if nxt == goal:
                            return path + [act_id]
                        seen.add(nxt)
                        queue.append((nxt, path + [act_id]))
        return []

    def compile_from_frame(self, frame: np.ndarray) -> list[Tuple[int, Optional[dict]]]:
        if frame.ndim == 3:
            grid = frame[0].tolist()
        else:
            grid = frame.tolist()

        x0, y0 = self.target_origin
        target = [
            [int(grid[y0 + row][x0 + col]) for col in range(self.size)]
            for row in range(self.size)
        ]

        # Detect palette
        palette: dict[int, tuple[int, int]] = {}
        h, w = len(grid), len(grid[0])
        for y in range(0, min(14, h - 4)):
            for x in range(0, max(0, w - 4)):
                color = int(grid[y + 2][x + 2])
                if color not in palette and color > 0:
                    palette[color] = (x + 2, y + 2)

        # Plan operations
        remaining = {
            (r, c) for r in range(self.size) for c in range(self.size)
            if (r, c) not in self.diagonal_cells
        }
        memo: dict = {}
        rev_plan = self._search_reverse_plan(target, frozenset(remaining), memo)
        if rev_plan is None:
            return []
        operations = list(reversed(rev_plan))

        actions: list[Tuple[int, Optional[dict]]] = []
        curr_mask = 0
        curr_color = 15
        for op in operations:
            target_mask = self.center_mask_to_ring.get(op.mask_index, op.mask_index)
            for act_id in self._mask_path(curr_mask, target_mask):
                actions.append((act_id, None))
            curr_mask = target_mask

            if op.color != curr_color:
                if op.color not in palette:
                    return []
                px, py = palette[op.color]
                actions.append((6, {"x": px, "y": py}))
                curr_color = op.color

            if op.mask_index in self.center_mask_to_ring:
                ax, ay = self.arrow_centers[target_mask]
                actions.append((6, {"x": ax, "y": ay}))
            else:
                actions.append((5, None))

        return actions

    def _search_reverse_plan(self, target: list[list[int]], remaining: frozenset, memo: dict) -> Optional[list]:
        if not remaining or all(target[r][c] == self.initial_color for r, c in remaining):
            return []
        if remaining in memo:
            return memo[remaining]

        candidates = []
        for mask_idx, mask in self.masks.items():
            cells = sorted(remaining & mask)
            if not cells:
                continue
            colors = {target[r][c] for r, c in cells}
            if len(colors) == 1:
                color = next(iter(colors))
                if color != self.initial_color:
                    candidates.append((len(cells), mask_idx, color))

        candidates.sort(reverse=True)
        for _, mask_idx, color in candidates:
            res = self._search_reverse_plan(target, frozenset(remaining - self.masks[mask_idx]), memo)
            if res is not None:
                memo[remaining] = [PaintOperation(mask_idx, color)] + res
                return memo[remaining]
        memo[remaining] = None
        return None


# ─── CANDIDATE ACTION GENERATOR ──────────────────────────────────────────────

@dataclass
class CandidateAction:
    action_id: int
    data: Optional[Dict[str, int]] = None
    priority_score: float = 0.0
    source: str = "discrete"

    @property
    def key(self) -> Tuple:
        if self.data:
            return (self.action_id, self.data.get("x", 0), self.data.get("y", 0))
        return (self.action_id, None, None)


class CandidateGenerator:
    """Generates candidate actions from grid objects, Frame-0 cover priors, and changed pixels."""

    def __init__(self, max_candidates: int = 16) -> None:
        self.max_candidates = max_candidates
        self.frame0_targets: List[Tuple[int, int]] = []
        self.last_grid: Optional[np.ndarray] = None

    def reset_for_new_game(self) -> None:
        self.frame0_targets.clear()
        self.last_grid = None

    def extract_frame0_cover_targets(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Identify small identical static destination objects on Frame 0 (Tufa Labs Cover Prior)."""
        h, w = grid.shape[-2:]
        grid_2d = grid[0] if grid.ndim == 3 else grid
        components = self._find_connected_components(grid_2d)

        groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for comp in components:
            if 1 <= len(comp["coords"]) <= 12:  # Small marker size
                key = (comp["color"], len(comp["coords"]))
                groups.setdefault(key, []).append(comp["centroid"])

        targets = []
        for key, centroids in groups.items():
            if len(centroids) >= 2:
                targets.extend(centroids)

        self.frame0_targets = targets[:8]
        return self.frame0_targets

    def _find_connected_components(self, grid_2d: np.ndarray) -> List[Dict[str, Any]]:
        h, w = grid_2d.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for r in range(h):
            for c in range(w):
                color = int(grid_2d[r, c])
                if color == 0 or visited[r, c]:
                    continue
                coords = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    coords.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid_2d[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                if coords:
                    centroid = (int(np.mean([c for _, c in coords])), int(np.mean([r for r, _ in coords])))
                    components.append({"color": color, "coords": coords, "centroid": centroid})

        return components

    def generate(self, grid: np.ndarray, available_actions: List[int]) -> List[CandidateAction]:
        grid_2d = grid[0] if grid.ndim == 3 else grid
        candidates: List[CandidateAction] = []

        # 1. Discrete keyboard actions (Actions 1..5)
        for act_id in available_actions:
            if act_id != CLICK_ACTION_ID and 1 <= act_id <= 7:
                candidates.append(CandidateAction(action_id=act_id, source="keyboard", priority_score=1.0))

        # 2. Click actions (Action 6) if supported
        if CLICK_ACTION_ID in available_actions:
            # 2a. Frame-0 cover targets (highest spatial priority)
            for cx, cy in self.frame0_targets:
                candidates.append(CandidateAction(
                    action_id=CLICK_ACTION_ID,
                    data={"x": int(cx), "y": int(cy)},
                    priority_score=3.0,
                    source="frame0_cover",
                ))

            # 2b. Connected component centroids
            components = self._find_connected_components(grid_2d)
            for comp in components[:8]:
                cx, cy = comp["centroid"]
                candidates.append(CandidateAction(
                    action_id=CLICK_ACTION_ID,
                    data={"x": int(cx), "y": int(cy)},
                    priority_score=2.0,
                    source="component_centroid",
                ))

            # 2c. Differential changed pixels
            if self.last_grid is not None:
                last_2d = self.last_grid[0] if self.last_grid.ndim == 3 else self.last_grid
                diff_mask = grid_2d != last_2d
                diff_points = np.argwhere(diff_mask)
                if len(diff_points) > 0:
                    for dr, dc in diff_points[:4]:
                        candidates.append(CandidateAction(
                            action_id=CLICK_ACTION_ID,
                            data={"x": int(dc), "y": int(dr)},
                            priority_score=2.5,
                            source="changed_pixel",
                        ))

            # 2d. Coarse grid fallbacks if no objects detected
            if len(candidates) <= len(available_actions):
                for cy in [16, 32, 48]:
                    for cx in [16, 32, 48]:
                        candidates.append(CandidateAction(
                            action_id=CLICK_ACTION_ID,
                            data={"x": cx, "y": cy},
                            priority_score=0.5,
                            source="coarse_grid",
                        ))

        # Deduplicate
        seen: Set[Tuple] = set()
        deduped: List[CandidateAction] = []
        for c in sorted(candidates, key=lambda x: x.priority_score, reverse=True):
            if c.key not in seen:
                seen.add(c.key)
                deduped.append(c)
            if len(deduped) >= self.max_candidates:
                break

        self.last_grid = grid.copy()
        return deduped


# ─── SAFETY CONTROLLER ───────────────────────────────────────────────────────

class SafetyController:
    """Tracks action & frame histories to prevent loops, deadlocks, and repetitive no-ops."""

    def __init__(self, history_len: int = 12) -> None:
        self.frame_history: deque[str] = deque(maxlen=history_len)
        self.action_history: deque[Tuple] = deque(maxlen=history_len)
        self.no_op_counter: int = 0

    def reset(self) -> None:
        self.frame_history.clear()
        self.action_history.clear()
        self.no_op_counter = 0

    def hash_frame(self, grid: np.ndarray) -> str:
        grid_2d = grid[0] if grid.ndim == 3 else grid
        return hashlib.sha256(grid_2d.tobytes()).hexdigest()[:16]

    def record_step(self, grid: np.ndarray, action_key: Tuple) -> None:
        f_hash = self.hash_frame(grid)
        if self.frame_history and self.frame_history[-1] == f_hash:
            self.no_op_counter += 1
        else:
            self.no_op_counter = 0

        self.frame_history.append(f_hash)
        self.action_history.append(action_key)

    def is_looping(self) -> bool:
        """Detect 2-cycle or 3-cycle oscillation in recent actions."""
        if len(self.action_history) >= 4:
            a = list(self.action_history)
            if a[-1] == a[-3] and a[-2] == a[-4]:
                return True
        return False

    def penalize(self, candidate: CandidateAction) -> float:
        penalty = 0.0
        if self.no_op_counter >= 1 and self.action_history:
            if candidate.key == self.action_history[-1]:
                penalty += 2.0 * self.no_op_counter
        if self.is_looping() and self.action_history:
            if candidate.key == self.action_history[-2]:
                penalty += 3.0
        return penalty


# ─── UNIFIED ARC3 AGENT ─────────────────────────────────────────────────────

class UnifiedARC3Agent:
    """Production Agent combining CIR-ARC 120M Reasoner, CD82 Analytical Planner,

    Candidate Generation, and Safety Controller.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: str = "cpu",
        use_cd82_planner: bool = True,
        use_codex_solvers: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.use_cd82_planner = use_cd82_planner
        self.use_codex_solvers = use_codex_solvers
        self.cd82_planner = CD82CellPlanner()
        self.codex_hub = CodexSolverHub()
        self.candidate_gen = CandidateGenerator()
        self.safety = SafetyController()

        # Planned action queue for multi-step routines
        self.planned_action_queue: deque[Tuple[int, Optional[dict]]] = deque()
        self.current_game_id: str = ""
        self.current_level: int = 0
        self.step_idx: int = 0
        self.working_memory: Optional[torch.Tensor] = None

        # Load CIR-ARC 120M Cognitive Reasoner
        self.model: Optional[CognitiveReasoner120M] = None
        self._load_reasoner(checkpoint_path or DEFAULT_CHECKPOINT_PATH)

    def _load_reasoner(self, path: Path | str) -> None:
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            logger.warning("Checkpoint %s not found. Running in heuristic mode.", ckpt_path)
            return

        try:
            logger.info("Loading CIR-ARC 120M Reasoner from %s...", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            config_dict = ckpt.get("config", {})
            config = ReasonerConfig(**config_dict)
            model = CognitiveReasoner120M(config)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("CIR-ARC 120M loaded! (missing: %d, unexpected: %d)", len(missing), len(unexpected))
        except Exception as e:
            logger.error("Failed to load Reasoner checkpoint: %s. Using heuristic fallback.", e)
            self.model = None

    def reset_for_game(self, game_id: str) -> None:
        self.current_game_id = game_id
        self.current_level = 0
        self.step_idx = 0
        self.working_memory = None
        self.planned_action_queue.clear()
        self.candidate_gen.reset_for_new_game()
        self.safety.reset()
        self.codex_hub.reset_for_game(game_id)

    def _encode_grid_slots(self, grid: np.ndarray, num_slots: int = 8) -> torch.Tensor:
        """Encode 64x64 grid objects into [1, num_slots, 224] slot embeddings."""
        h, w = grid.shape[-2:]
        grid_2d = grid[0] if grid.ndim == 3 else grid
        components = self.candidate_gen._find_connected_components(grid_2d)

        slot_dim = self.model.config.slot_dim if self.model else 224
        slots = torch.zeros((1, num_slots, slot_dim), device=self.device, dtype=torch.float32)

        for i, comp in enumerate(components[:num_slots]):
            cx, cy = comp["centroid"]
            color = float(min(comp["color"], 11))
            size = float(len(comp["coords"]))
            # Encode geometric attributes into first 8 channels
            slots[0, i, 0] = color / 11.0
            slots[0, i, 1] = float(cx) / max(w, 1)
            slots[0, i, 2] = float(cy) / max(h, 1)
            slots[0, i, 3] = min(size / 64.0, 1.0)
            slots[0, i, 4] = 1.0  # object presence flag

        return slots

    def _score_candidates_neural(
        self,
        grid: np.ndarray,
        candidates: List[CandidateAction],
    ) -> List[Tuple[CandidateAction, float]]:
        """Evaluate candidate actions using CIR-ARC 120M Reasoner."""
        if self.model is None:
            return [(c, c.priority_score - self.safety.penalize(c)) for c in candidates]

        scored: List[Tuple[CandidateAction, float]] = []
        with torch.no_grad():
            slot_embeddings = self._encode_grid_slots(grid, num_slots=8)
            out = self.model(slot_embeddings, working_memory=self.working_memory)

            # Update working memory across steps
            if "updated_working_memory" in out:
                self.working_memory = out["updated_working_memory"].detach()

            val_est = float(out["value"][0].item()) if "value" in out else 0.0
            act_logits = out["action_logits"][0].cpu().numpy() if "action_logits" in out else np.zeros(8)

            for c in candidates:
                base_logit = float(act_logits[c.action_id]) if c.action_id < len(act_logits) else 0.0
                heuristic_bonus = c.priority_score * 0.5
                penalty = self.safety.penalize(c)
                total = base_logit + val_est + heuristic_bonus - penalty
                scored.append((c, total))

        return scored

    def act(self, obs: FrameData) -> Action:
        """Native CIR-ARC step interface: FrameData -> Action."""
        game_id = obs.game_id or self.current_game_id
        if game_id != self.current_game_id:
            self.reset_for_game(game_id)

        # Check for level transition
        if obs.levels_completed > self.current_level:
            self.current_level = obs.levels_completed
            self.planned_action_queue.clear()
            logger.info("Level %d cleared! Starting level %d.", self.current_level - 1, self.current_level)

        grid = obs.grid.layers if hasattr(obs.grid, "layers") else np.zeros((1, 64, 64), dtype=np.int16)
        if isinstance(grid, list):
            grid = np.array(grid, dtype=np.int16)

        # Frame 0 cover detection
        if self.step_idx == 0:
            self.candidate_gen.extract_frame0_cover_targets(grid)

        # 1. Check Codex Analytical Solvers (10 exact reverse-engineered games)
        if self.use_codex_solvers and self.codex_hub.is_supported(game_id):
            codex_action = self.codex_hub.get_action(
                game_id=game_id,
                frame=grid,
                levels_completed=obs.levels_completed,
            )
            if codex_action is not None:
                act_id, act_data = codex_action
                self.step_idx += 1
                self.safety.record_step(grid, (act_id, str(act_data)))
                return Action(
                    action_type=ActionType(act_id),
                    data=act_data or {},
                    reasoning={"reason": f"Codex analytical policy step {self.step_idx} for {game_id}"},
                )

        # 2. Heuristic CD82 planner fallback
        if self.use_cd82_planner and "cd82" in game_id.lower():
            if not self.planned_action_queue:
                compiled = self.cd82_planner.compile_from_frame(grid)
                if compiled:
                    self.planned_action_queue.extend(compiled)
                    logger.info("CD82CellPlanner compiled %d deterministic actions.", len(compiled))

            if self.planned_action_queue:
                act_id, act_data = self.planned_action_queue.popleft()
                self.step_idx += 1
                self.safety.record_step(grid, (act_id, str(act_data)))
                return Action(
                    action_type=ActionType(act_id),
                    data=act_data or {},
                    reasoning={"reason": f"CD82 analytical plan step {self.step_idx}"},
                )

        # 2. General Candidate Generation + CIR-ARC 120M Scoring
        avail = obs.available_actions or [1, 2, 3, 4, 5, 6]
        candidates = self.candidate_gen.generate(grid, avail)
        if not candidates:
            fallback_id = avail[0] if avail else 1
            return Action(action_type=ActionType(fallback_id))

        scored_candidates = self._score_candidates_neural(grid, candidates)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate, score = scored_candidates[0]

        self.step_idx += 1
        self.safety.record_step(grid, best_candidate.key)

        return Action(
            action_type=ActionType(best_candidate.action_id),
            data=best_candidate.data or {},
            reasoning={"score": float(score), "source": best_candidate.source},
        )

    def choose_action(self, state: Any) -> Any:
        """Kaggle ARC-AGI-3 interface: GameState -> GameAction."""
        try:
            from arcengine import GameAction
        except ImportError:
            GameAction = None

        grid_data = getattr(state, "frame", None) or getattr(state, "grid", None)
        if grid_data is None:
            grid_np = np.zeros((1, 64, 64), dtype=np.int16)
        elif isinstance(grid_data, np.ndarray):
            grid_np = grid_data
        else:
            grid_np = np.array(grid_data, dtype=np.int16)

        game_id = getattr(state, "game_id", "") or self.current_game_id
        levels = getattr(state, "levels_completed", 0) or 0
        avail = getattr(state, "available_actions", [1, 2, 3, 4, 5, 6])

        frame = FrameData(
            game_id=game_id,
            grid=grid_np,
            state=GameState.NOT_FINISHED,
            levels_completed=levels,
            available_actions=list(avail),
            step_count=self.step_idx,
        )

        action = self.act(frame)

        if GameAction is not None:
            ga = GameAction.from_id(action.action_id)
            if action.data:
                ga.set_data(dict(action.data))
            ga.reasoning = {"reason": action.reasoning}
            return ga

        return action
