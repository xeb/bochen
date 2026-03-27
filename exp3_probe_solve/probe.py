"""Systematic diagnostic probe battery."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcengine import GameAction
from shared.env_wrapper import ArcEnv, SIMPLE_ACTIONS
from shared.perception import find_background_color, connected_components
from exp3_probe_solve.memory import CausalMemory


def run_probe_battery(env: ArcEnv, memory: CausalMemory, game_id: str) -> list[dict]:
    """Run systematic diagnostic probes. Returns action-effect log."""
    log = []

    def probe(action, data=None, description=""):
        """Execute one probe from fresh state, record effect."""
        grid_before, _, score_before, _ = env.reset()
        grid_after, state, score_after, _ = env.step(action, data=data)

        delta = int(np.sum(grid_before != grid_after))
        effect = {
            "cells_changed": delta,
            "score_delta": float(score_after - score_before),
            "state_after": state,
            "description": description,
        }
        memory.store(grid_before, str(action), effect, game_id)

        entry = {
            "action": str(action),
            "data": data,
            "effect": effect,
            "description": description,
        }
        log.append(entry)
        return grid_after, state, effect

    # --- Probe 1: Try each simple action from clean state ---
    for i, action in enumerate(SIMPLE_ACTIONS):
        probe(action, description=f"Test ACTION{i+1} from initial state")

    # --- Probe 2: Click on each distinct object with ACTION6 ---
    grid, _, _, _ = env.reset()
    bg = find_background_color(grid)
    objects = connected_components(grid, bg)

    for obj in objects[:8]:  # cap at 8 objects
        cx, cy = int(obj["center"][0]), int(obj["center"][1])
        probe(
            GameAction.ACTION6,
            data={"x": cx, "y": cy},
            description=f"Click object color={obj['color']} at ({cx},{cy})"
        )

    # --- Probe 3: Click on empty space ---
    h, w = grid.shape
    bg_positions = np.argwhere(grid == bg)
    if len(bg_positions) > 0:
        by, bx = bg_positions[len(bg_positions) // 2]
        probe(
            GameAction.ACTION6,
            data={"x": int(bx), "y": int(by)},
            description="Click on background/empty space"
        )

    # --- Probe 4: Test undo (ACTION7) ---
    grid_before, _, _, _ = env.reset()
    env.step(GameAction.ACTION1)
    grid_after_action, _, _, _ = env.step(GameAction.ACTION7)

    undo_delta = int(np.sum(grid_before != grid_after_action))
    undo_effect = {
        "cells_changed": undo_delta,
        "score_delta": 0.0,
        "state_after": "NOT_FINISHED",
        "description": "ACTION7 undo after ACTION1",
        "undo_restored": undo_delta == 0,  # did undo fully restore?
    }
    memory.store(grid_before, "ACTION7_UNDO_TEST", undo_effect, game_id)
    log.append({"action": "ACTION7", "effect": undo_effect,
                "description": "Undo test after ACTION1"})

    # --- Probe 5: Repeat same action twice (consistency check) ---
    grid1, _, _, _ = env.reset()
    grid2, _, _, _ = env.step(GameAction.ACTION1)
    grid3, _, _, _ = env.step(GameAction.ACTION1)

    delta1 = int(np.sum(grid1 != grid2))
    delta2 = int(np.sum(grid2 != grid3))
    consistency_effect = {
        "first_delta": delta1,
        "second_delta": delta2,
        "consistent": delta1 == delta2,
        "cumulative": delta2 > 0 and delta1 > 0,
    }
    log.append({"action": "ACTION1_REPEAT", "effect": consistency_effect,
                "description": "Repeat ACTION1 twice for consistency check"})

    return log
