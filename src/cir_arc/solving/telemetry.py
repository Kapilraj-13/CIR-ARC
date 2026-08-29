from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    event_name: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class AgentOpsTelemetry:
    """Telemetry tracker for agent decisions, step latency, tool executions, and session traces."""

    def __init__(self, agent_name: str = "CIR-ARC-Agent") -> None:
        self.agent_name = agent_name
        self.events: List[TelemetryEvent] = []
        self.start_time = time.time()

    def log_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = TelemetryEvent(
            event_name=event_name,
            timestamp=time.time(),
            data=payload or {},
        )
        self.events.append(event)
        logger.debug("[Telemetry] %s: %s", event_name, payload)

    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "agent_name": self.agent_name,
            "total_events": len(self.events),
            "elapsed_seconds": round(elapsed, 3),
            "events": [
                {"name": e.event_name, "time": round(e.timestamp - self.start_time, 3), "data": e.data}
                for e in self.events[-50:]  # Keep last 50
            ],
        }
