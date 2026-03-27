"""Wrapper around arc_agi environments for consistent interface."""

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction


# Map string names to GameAction enums
ACTION_MAP = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}

SIMPLE_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5,
]

ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]


def parse_action(name: str) -> GameAction:
    """Convert string action name to GameAction enum."""
    name = name.strip().upper()
    if name in ACTION_MAP:
        return ACTION_MAP[name]
    # Try stripping prefix
    for key, val in ACTION_MAP.items():
        if key in name:
            return val
    return GameAction.ACTION1


class ArcEnv:
    """Thin wrapper for consistent grid access and state checking."""

    def __init__(self, game_id: str, offline: bool = True, render: bool = False):
        mode = OperationMode.OFFLINE if offline else OperationMode.ONLINE
        self.arcade = Arcade(operation_mode=mode)
        self.game_id = game_id
        render_mode = "terminal" if render else None
        self.env = self.arcade.make(game_id, render_mode=render_mode)
        self._last_obs = None

    def reset(self):
        """Reset environment and return grid as numpy array."""
        self._last_obs = self.env.reset()
        return self._extract()

    def step(self, action: GameAction, data: dict = None):
        """Take an action, return (grid, state, score, obs)."""
        if data:
            self._last_obs = self.env.step(action, data=data)
        else:
            self._last_obs = self.env.step(action)
        return self._extract()

    def _extract(self):
        """Extract grid, state, score from observation."""
        obs = self._last_obs
        grid = np.array(obs.grid) if hasattr(obs, 'grid') else np.array(obs)

        state = "NOT_FINISHED"
        if hasattr(obs, 'state'):
            state = str(obs.state)
            # Normalize state strings
            s = state.upper()
            if "WIN" in s:
                state = "WIN"
            elif "OVER" in s or "GAME_OVER" in s:
                state = "GAME_OVER"
            else:
                state = "NOT_FINISHED"

        score = getattr(obs, 'score', 0.0)
        return grid, state, score, obs

    @property
    def is_won(self) -> bool:
        if self._last_obs is None:
            return False
        s = str(getattr(self._last_obs, 'state', ''))
        return "WIN" in s.upper()

    @property
    def is_over(self) -> bool:
        if self._last_obs is None:
            return False
        s = str(getattr(self._last_obs, 'state', ''))
        return "OVER" in s.upper() or "GAME_OVER" in s.upper()

    @property
    def is_done(self) -> bool:
        return self.is_won or self.is_over
