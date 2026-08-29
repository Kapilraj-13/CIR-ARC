from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cir_arc.environment.actions import DIRECTION_VECTORS, Action
from cir_arc.environment.frame import FrameData
from cir_arc.environment.state_delta import StateDelta, compute_state_delta
from cir_arc.hypothesis.delta_mapper import DeltaMapper, SymbolicDelta
from cir_arc.hypothesis.transition_grammar import TransitionRule

logger = logging.getLogger(__name__)


class HypothesisInductionEngine:
    """Induces, corroborates, and prunes causal transition rules from interactive probe traces."""

    def __init__(self) -> None:
        self.rules: Dict[str, TransitionRule] = {}
        self.observations: List[Tuple[FrameData, Action, FrameData, StateDelta]] = []

    def observe_transition(
        self,
        before: FrameData,
        action: Action,
        after: FrameData,
    ) -> Optional[TransitionRule]:
        delta = compute_state_delta(before, after, action)
        self.observations.append((before, action, after, delta))

        sym_delta = DeltaMapper.map_delta(delta)
        aid = action.action_id

        # 1. Movement rule induction
        if aid in DIRECTION_VECTORS and sym_delta.movement_vectors:
            mv = sym_delta.movement_vectors[0]
            rule_id = f"RULE_MOVE_{aid}"
            dr, dc = int(round(mv["dr"])), int(round(mv["dc"]))

            if rule_id not in self.rules:
                rule = TransitionRule(
                    rule_id=rule_id,
                    action_trigger=aid,
                    condition_type="ALWAYS",
                    effect_type="TRANSLATE_PLAYER",
                    params={"dr": dr, "dc": dc, "color": mv["color"]},
                    confidence=1.0,
                    support_count=1,
                )
                self.rules[rule_id] = rule
                return rule
            else:
                rule = self.rules[rule_id]
                if (rule.params.get("dr"), rule.params.get("dc")) == (dr, dc):
                    rule.support_count += 1
                    rule.confidence = float(rule.support_count) / (rule.support_count + rule.refutation_count)
                else:
                    rule.refutation_count += 1
                    rule.confidence = float(rule.support_count) / (rule.support_count + rule.refutation_count)
                return rule

        # 2. Collectible / interaction rule induction
        if aid == 5 and sym_delta.destroyed_archetypes:
            dest = sym_delta.destroyed_archetypes[0]
            rule_id = f"RULE_COLLECT_C{dest['color']}"
            if rule_id not in self.rules:
                rule = TransitionRule(
                    rule_id=rule_id,
                    action_trigger=5,
                    condition_type="OBJECT_PRESENT",
                    effect_type="REMOVE_OBJECT",
                    params={"target_color": dest["color"]},
                    confidence=1.0,
                    support_count=1,
                )
                self.rules[rule_id] = rule
                return rule
            else:
                rule = self.rules[rule_id]
                rule.support_count += 1
                rule.confidence = float(rule.support_count) / (rule.support_count + rule.refutation_count)
                return rule

        return None

    def get_best_rule_for_action(self, action_id: int) -> Optional[TransitionRule]:
        candidates = [r for r in self.rules.values() if r.action_trigger == action_id]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.confidence, r.support_count))

    def get_all_rules(self) -> List[TransitionRule]:
        return sorted(self.rules.values(), key=lambda r: (-r.confidence, -r.support_count))

    def prune_refuted(self, min_confidence: float = 0.4) -> None:
        self.rules = {
            rid: r for rid, r in self.rules.items()
            if r.confidence >= min_confidence
        }
