"""Macro action abstractions and registry for compound navigation and interaction sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from cir_arc.environment.actions import Action


@dataclass
class MacroAction:
    """A named composite action sequence."""
    macro_id: str
    description: str
    actions: List[Action] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


class MacroRegistry:
    """Maintains reusable macro action patterns."""

    def __init__(self) -> None:
        self.macros: Dict[str, MacroAction] = {}

    def register_macro(self, macro_id: str, description: str, actions: List[Action]) -> MacroAction:
        m = MacroAction(macro_id=macro_id, description=description, actions=actions)
        self.macros[macro_id] = m
        return m

    def get_macro(self, macro_id: str) -> Optional[MacroAction]:
        return self.macros.get(macro_id)
