"""Collect transition data from random play for world model training."""

import sys
import os
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcengine import GameAction
from shared.env_wrapper import ArcEnv, SIMPLE_ACTIONS, ALL_ACTIONS
from shared.perception import one_hot_grid


def collect_transitions(game_id: str, num_steps: int = 50000,
                        save_dir: str = None) -> str:
    """Run random agent, collect (grid, action, next_grid) transitions.
    Returns path to saved .npz file.
    """
    save_dir = save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{game_id}_transitions.npz")

    env = ArcEnv(game_id, offline=True, render=False)
    grid, state, score, obs = env.reset()
    h, w = grid.shape

    grids_before = []
    actions = []
    grids_after = []
    collected = 0

    while collected < num_steps:
        # Random action
        action = random.choice(ALL_ACTIONS)
        action_idx = ALL_ACTIONS.index(action)
        data = None
        if action == GameAction.ACTION6:
            data = {"x": random.randint(0, w - 1), "y": random.randint(0, h - 1)}

        grid_before = grid.copy()
        grid, state, score, obs = env.step(action, data=data)

        grids_before.append(grid_before)
        actions.append(action_idx)
        grids_after.append(grid.copy())
        collected += 1

        if state in ("WIN", "GAME_OVER"):
            grid, state, score, obs = env.reset()

        if collected % 10000 == 0:
            print(f"  [exp2 collect] {game_id}: {collected}/{num_steps} transitions")

    np.savez_compressed(
        save_path,
        grids_before=np.array(grids_before, dtype=np.int8),
        actions=np.array(actions, dtype=np.int8),
        grids_after=np.array(grids_after, dtype=np.int8),
    )
    print(f"  [exp2 collect] Saved {collected} transitions to {save_path}")
    return save_path
