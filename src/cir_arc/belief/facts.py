"""Explicit provenance tracking and epistemic facts for ARC-AGI-3 belief states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class Provenance(str, Enum):
    """Epistemic status of a piece of knowledge."""
    FACT = "FACT"                  # Ground truth invariant (e.g. grid bounds, player color)
    OBSERVED = "OBSERVED"          # Directly witnessed in visual frames
    INFERRED = "INFERRED"          # Symbolically deduced with high confidence
    PREDICTED = "PREDICTED"        # Produced by world-model forward simulation
    HYPOTHESIS = "HYPOTHESIS"      # Tentative induction pending verification


class FactType(str, Enum):
    """Categorical types of epistemic facts."""
    OBJECT_EXISTS = "OBJECT_EXISTS"
    OBJECT_COLOR = "OBJECT_COLOR"
    OBJECT_POSITION = "OBJECT_POSITION"
    OBJECT_ROLE = "OBJECT_ROLE"               # e.g. "player", "key", "door", "goal", "wall"
    PASSABILITY = "PASSABILITY"               # Is coordinate/color passable?
    INTERACTION_RULE = "INTERACTION_RULE"     # What happens when player touches object?
    GOAL_CANDIDATE = "GOAL_CANDIDATE"         # Is this state or object terminal win?
    RESOURCE_AMOUNT = "RESOURCE_AMOUNT"       # Inventory or score


@dataclass
class Fact:
    """An individual fact with explicit provenance and confidence."""
    fact_id: str
    fact_type: FactType
    subject: str                               # e.g. "obj_9", "pos_(3,4)", "color_5"
    predicate: str                             # e.g. "is_passable", "is_key_for_door"
    value: Any                                 # e.g. True, False, "door_8", (5, 5)
    provenance: Provenance = Provenance.OBSERVED
    confidence: float = 1.0                    # [0.0, 1.0]
    evidence_count: int = 1
    contradiction_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """Returns True if the fact is established as a reliable fact or repeated observation."""
        return (
            self.provenance in (Provenance.FACT, Provenance.OBSERVED)
            and self.confidence >= 0.8
            and self.contradiction_count == 0
        )

    def reinforce(self, confidence_boost: float = 0.1) -> None:
        """Increases confidence upon consistent re-observation."""
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + confidence_boost)

    def contradict(self, penalty: float = 0.3) -> None:
        """Decreases confidence upon counterexample observation."""
        self.contradiction_count += 1
        self.confidence = max(0.0, self.confidence - penalty)
        if self.confidence < 0.3 and self.provenance != Provenance.FACT:
            self.provenance = Provenance.HYPOTHESIS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": str(self.value),
            "provenance": self.provenance.value,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
        }


class FactSet:
    """Container managing a consistent set of epistemic facts with lookup by subject/predicate."""

    def __init__(self) -> None:
        self.facts: Dict[str, Fact] = {}

    def add_or_update(
        self,
        fact_type: FactType,
        subject: str,
        predicate: str,
        value: Any,
        provenance: Provenance = Provenance.OBSERVED,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        fact_id = f"{fact_type.value}:{subject}:{predicate}"
        if fact_id in self.facts:
            existing = self.facts[fact_id]
            if existing.value == value:
                existing.reinforce()
            else:
                existing.contradict()
                if existing.confidence < 0.2:
                    existing.value = value
                    existing.provenance = provenance
                    existing.confidence = confidence
                    existing.contradiction_count = 0
            return existing

        new_fact = Fact(
            fact_id=fact_id,
            fact_type=fact_type,
            subject=subject,
            predicate=predicate,
            value=value,
            provenance=provenance,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.facts[fact_id] = new_fact
        return new_fact

    def get_fact(self, fact_type: FactType, subject: str, predicate: str) -> Optional[Fact]:
        fact_id = f"{fact_type.value}:{subject}:{predicate}"
        return self.facts.get(fact_id)

    def query(
        self,
        fact_type: Optional[FactType] = None,
        subject: Optional[str] = None,
        provenance: Optional[Provenance] = None,
        min_confidence: float = 0.0,
    ) -> List[Fact]:
        results = []
        for fact in self.facts.values():
            if fact_type is not None and fact.fact_type != fact_type:
                continue
            if subject is not None and fact.subject != subject:
                continue
            if provenance is not None and fact.provenance != provenance:
                continue
            if fact.confidence < min_confidence:
                continue
            results.append(fact)
        return results

    def get_known_passable_colors(self) -> Set[int]:
        """Returns set of colors confirmed to be passable."""
        passable: Set[int] = set()
        for f in self.query(fact_type=FactType.PASSABILITY, min_confidence=0.5):
            if f.value is True and f.subject.startswith("color_"):
                try:
                    c = int(f.subject.replace("color_", ""))
                    passable.add(c)
                except ValueError:
                    pass
        return passable

    def get_known_blocked_colors(self) -> Set[int]:
        """Returns set of colors confirmed to be non-passable obstacles."""
        blocked: Set[int] = set()
        for f in self.query(fact_type=FactType.PASSABILITY, min_confidence=0.5):
            if f.value is False and f.subject.startswith("color_"):
                try:
                    c = int(f.subject.replace("color_", ""))
                    blocked.add(c)
                except ValueError:
                    pass
        return blocked

    def __len__(self) -> int:
        return len(self.facts)
