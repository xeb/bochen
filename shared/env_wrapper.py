"""Wrapper around arc_agi environments for consistent interface."""

import time
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from arcengine.enums import GameState

# Global rate limiter: 300 RPM target (well under 600 RPM limit)
_MIN_INTERVAL = 60.0 / 300
_last_api_call = 0.0


def _rate_limit():
    global _last_api_call
    now = time.monotonic()
    elapsed = now - _last_api_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_api_call = time.monotonic()


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

# Map action int IDs to GameAction enums
ACTION_INT_MAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
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


def extract_grid(obs) -> np.ndarray:
    """Extract 2D grid array from a FrameDataRaw observation.

    obs.frame is a list that becomes shape (1, H, W) — we take [0] to get (H, W).
    """
    frame = np.array(obs.frame)
    if frame.ndim == 3:
        return frame[0]  # (1, H, W) -> (H, W)
    return frame


def extract_state(obs) -> str:
    """Normalize obs.state (GameState enum) to a simple string."""
    state = getattr(obs, 'state', None)
    if state is None:
        return "NOT_FINISHED"
    if state == GameState.WIN:
        return "WIN"
    if state == GameState.GAME_OVER:
        return "GAME_OVER"
    s = str(state).upper()
    if "WIN" in s:
        return "WIN"
    if "OVER" in s:
        return "GAME_OVER"
    return "NOT_FINISHED"


def get_available_actions(obs) -> list[GameAction]:
    """Get list of available GameAction enums from observation."""
    raw = getattr(obs, 'available_actions', None)
    if raw is None:
        return SIMPLE_ACTIONS  # fallback
    return [ACTION_INT_MAP[i] for i in raw if i in ACTION_INT_MAP]


class ArcEnv:
    """Thin wrapper for consistent grid access and state checking."""

    def __init__(self, game_id: str, offline: bool = False, render: bool = False):
        # Default to ONLINE since OFFLINE needs local game files
        mode = OperationMode.OFFLINE if offline else OperationMode.ONLINE
        self.arcade = Arcade(operation_mode=mode)
        self.game_id = game_id
        render_mode = "terminal" if render else None
        self.env = self.arcade.make(game_id, render_mode=render_mode)
        self._last_obs = None
        self.available_actions = SIMPLE_ACTIONS

    def reset(self):
        """Reset environment. Returns (grid, state, score, raw_obs)."""
        _rate_limit()
        self._last_obs = self.env.reset()
        return self._extract()

    def step(self, action: GameAction, data: dict = None):
        """Take an action. Returns (grid, state, score, raw_obs)."""
        _rate_limit()
        try:
            result = self.env.step(action, data=data)
            if result is not None:
                self._last_obs = result
        except Exception as e:
            print(f"  [env] step error: {e}")
            # Return last known state
        return self._extract()

    def _extract(self):
        """Extract (grid, state, score, raw_obs) from observation."""
        obs = self._last_obs
        if obs is None:
            # Return safe defaults
            import numpy as np
            return np.zeros((64, 64), dtype=np.int8), "GAME_OVER", 0, None
        grid = extract_grid(obs)
        state = extract_state(obs)
        score = getattr(obs, 'levels_completed', 0) or 0
        self.available_actions = get_available_actions(obs)
        return grid, state, score, obs

    @property
    def is_won(self) -> bool:
        if self._last_obs is None:
            return False
        return extract_state(self._last_obs) == "WIN"

    @property
    def is_over(self) -> bool:
        if self._last_obs is None:
            return False
        return extract_state(self._last_obs) == "GAME_OVER"

    @property
    def is_done(self) -> bool:
        return self.is_won or self.is_over
