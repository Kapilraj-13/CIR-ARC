from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from cir_arc.core.grid import Grid


@dataclass
class GridPair:
    input: Grid
    output: Optional[Grid] = None  # None when output is withheld (test eval)

    def has_output(self) -> bool:
        return self.output is not None


@dataclass
class ArcTask:
    task_id: str
    train_pairs: List[GridPair]
    test_pairs: List[GridPair]

    # Metadata â€” populated for synthetic tasks, None for official ARC
    source: str = "unknown"              # "official", "rearC", "synthetic", "community"
    rule_type: Optional[str] = None      # e.g. "reflect_horizontal"
    rule_params: Optional[Dict] = None   # e.g. {"axis": "horizontal"}
    difficulty: Optional[int] = None     # 1=single rule, 2=two-rule, etc.

    # Auto-computed
    content_hash: str = field(init=False)

    def __post_init__(self):
        self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Stable hash of grid contents â€” used for leakage detection."""
        h = hashlib.sha256()
        for pair in self.train_pairs + self.test_pairs:
            h.update(pair.input.data.tobytes())
            if pair.output is not None:
                h.update(pair.output.data.tobytes())
        return h.hexdigest()[:16]

    @property
    def n_train(self) -> int:
        return len(self.train_pairs)

    @property
    def n_test(self) -> int:
        return len(self.test_pairs)

    # â”€â”€ Serialization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (superset of official ARC JSON format)."""
        return {
            "task_id": self.task_id,
            "source": self.source,
            "rule_type": self.rule_type,
            "rule_params": self.rule_params,
            "difficulty": self.difficulty,
            "content_hash": self.content_hash,
            "train": [
                {
                    "input": pair.input.to_list(),
                    "output": pair.output.to_list() if pair.output else None
                }
                for pair in self.train_pairs
            ],
            "test": [
                {
                    "input": pair.input.to_list(),
                    "output": pair.output.to_list() if pair.output else None
                }
                for pair in self.test_pairs
            ]
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArcTask:
        def load_pair(p: dict) -> GridPair:
            inp = Grid.from_list(p["input"])
            out = Grid.from_list(p["output"]) if p.get("output") else None
            return GridPair(input=inp, output=out)

        return cls(
            task_id=d.get("task_id", "unknown"),
            source=d.get("source", "unknown"),
            rule_type=d.get("rule_type"),
            rule_params=d.get("rule_params"),
            difficulty=d.get("difficulty"),
            train_pairs=[load_pair(p) for p in d["train"]],
            test_pairs=[load_pair(p) for p in d["test"]],
        )

    @classmethod
    def load(cls, path: Path) -> ArcTask:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_official_arc_dir(
        cls, dir_path: Path, source: str = "official"
    ) -> List[ArcTask]:
        """
        Load all tasks from an official ARC directory
        (training/ or evaluation/ subfolder).
        """
        tasks = []
        for json_path in sorted(Path(dir_path).glob("*.json")):
            with open(json_path) as f:
                raw = json.load(f)
            task = cls(
                task_id=json_path.stem,
                source=source,
                train_pairs=[
                    GridPair(
                        Grid.from_list(p["input"]),
                        Grid.from_list(p["output"])
                    )
                    for p in raw["train"]
                ],
                test_pairs=[
                    GridPair(
                        Grid.from_list(p["input"]),
                        Grid.from_list(p["output"]) if "output" in p else None
                    )
                    for p in raw["test"]
                ],
            )
            tasks.append(task)
        return tasks
