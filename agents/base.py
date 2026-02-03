from __future__ import annotations

from typing import Protocol

import numpy as np

from core.types import GameState


class Agent(Protocol):
    """Protocol that all snake agents must satisfy."""

    agent_type: str

    def act(self, state: GameState) -> int:
        """Return an action index (0-5) given the current game state."""
        ...
