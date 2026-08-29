from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records gameplay frames and actions to JSONL format compatible with ARC-AGI-3 standards."""

    def __init__(self, game_id: str, output_path: Optional[str] = None) -> None:
        self.game_id = game_id
        self.timestamp = int(time.time())
        self.output_path = output_path or f"recordings/{game_id}_{self.timestamp}.recording.jsonl"
        self.records: List[Dict[str, Any]] = []

    def record(self, item: Union[FrameData, Action, Dict[str, Any]]) -> None:
        """Record a single frame, action, or metadata item."""
        if isinstance(item, FrameData):
            entry = {
                "type": "frame",
                "timestamp": time.time(),
                "data": item.to_dict(),
            }
        elif isinstance(item, Action):
            entry = {
                "type": "action",
                "timestamp": time.time(),
                "data": item.to_dict(),
            }
        elif isinstance(item, dict):
            entry = {
                "type": "event",
                "timestamp": time.time(),
                "data": item,
            }
        else:
            entry = {
                "type": "custom",
                "timestamp": time.time(),
                "data": str(item),
            }

        self.records.append(entry)

    def flush(self) -> str:
        """Write recorded entries to output JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec) + "\n")
        logger.info("Saved %d records to %s", len(self.records), self.output_path)
        return self.output_path

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self.records)
