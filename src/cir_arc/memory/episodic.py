"""Persistent Episodic Memory tracking interaction traces, rules, and game outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

from cir_arc.environment.actions import Action
from cir_arc.environment.frame import FrameData, GameState


@dataclass
class Episode:
    """Complete record of an interactive session on an ARC-AGI-3 puzzle."""
    episode_id: str
    game_id: str
    initial_frame: FrameData
    actions: List[Action] = field(default_factory=list)
    state_history: List[str] = field(default_factory=list)
    discovered_rules: List[Dict[str, Any]] = field(default_factory=list)
    goal: Optional[Dict[str, Any]] = None
    outcome: GameState = GameState.NOT_FINISHED
    total_steps: int = 0
    replayable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "game_id": self.game_id,
            "total_steps": self.total_steps,
            "outcome": self.outcome.value,
            "actions_count": len(self.actions),
            "discovered_rules_count": len(self.discovered_rules),
        }


class EpisodicMemory:
    """Manages long-term storage and retrieval of interactive puzzle episodes."""

    def __init__(self) -> None:
        self.episodes: Dict[str, Episode] = {}

    def store_episode(self, episode: Episode) -> None:
        self.episodes[episode.episode_id] = episode

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        return self.episodes.get(episode_id)

    def find_similar_episodes(self, game_id: str) -> List[Episode]:
        """Finds previously solved episodes with identical or matching game_id prefix."""
        matches = []
        for ep in self.episodes.values():
            if ep.game_id == game_id or ep.game_id.split("_")[0] in game_id:
                matches.append(ep)
        return matches
