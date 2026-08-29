from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from cir_arc.environment.actions import Action, ActionType
from cir_arc.environment.base import BaseEnvironment
from cir_arc.environment.frame import FrameData

logger = logging.getLogger(__name__)


class PlaybackAgent:
    """Plays back recorded action traces against an environment for deterministic verification."""

    def __init__(self, recording_path: str) -> None:
        self.recording_path = recording_path
        self.actions: List[Action] = []
        self.raw_records: List[Dict[str, Any]] = []
        self._load_recording()

    def _load_recording(self) -> None:
        self.actions = []
        self.raw_records = []
        try:
            with open(self.recording_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self.raw_records.append(record)

            has_explicit_actions = any(r.get("type") == "action" for r in self.raw_records)
            for record in self.raw_records:
                rec_type = record.get("type")
                data = record.get("data", {})
                if has_explicit_actions:
                    if rec_type == "action" and "id" in data:
                        action = Action.from_id(
                            action_id=data["id"],
                            data=data.get("data", {}),
                            reasoning=data.get("reasoning", {}),
                        )
                        self.actions.append(action)
                else:
                    if rec_type == "frame" and "action_input" in data and data["action_input"]:
                        ainput = data["action_input"]
                        aid = ainput.get("id") or ainput.get("action_id", 0)
                        if isinstance(aid, str):
                            try:
                                aid = int(ActionType[aid])
                            except KeyError:
                                aid = 0
                        if int(aid) != 0:
                            action = Action.from_id(
                                action_id=int(aid),
                                data=ainput.get("data", {}),
                                reasoning=ainput.get("reasoning", {}),
                            )
                            self.actions.append(action)
            logger.info("Loaded %d actions from recording %s", len(self.actions), self.recording_path)
        except Exception as e:
            logger.error("Failed to load recording %s: %s", self.recording_path, e)

    def replay_all(self, env: BaseEnvironment) -> List[FrameData]:
        """Replay all recorded actions sequentially on the environment."""
        frames: List[FrameData] = []
        initial_frame = env.reset()
        frames.append(initial_frame)

        for action in self.actions:
            frame = env.step(action)
            frames.append(frame)
            if frame.is_terminal:
                break

        return frames
