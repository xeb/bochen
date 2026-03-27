"""Solver phase: use probe results and memory to construct and execute plans."""

import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcengine import GameAction
from shared.env_wrapper import ArcEnv, SIMPLE_ACTIONS, ALL_ACTIONS, parse_action
from shared.perception import find_background_color, connected_components
from exp3_probe_solve.memory import CausalMemory
from exp3_probe_solve.config import MAX_SOLVE_STEPS


def analyze_probes(probe_log: list[dict]) -> dict:
    """Analyze probe results to build an action-effect model."""
    model = {
        "effective_actions": [],   # actions that changed the grid
        "ineffective_actions": [], # actions that did nothing
        "score_actions": [],       # actions that changed the score
        "best_action": None,       # action with highest score delta
        "undo_available": False,
        "action_consistent": True,
        "click_targets": [],       # (x, y) positions that had effects
    }

    best_score_delta = -float("inf")

    for entry in probe_log:
        effect = entry.get("effect", {})
        action = entry.get("action", "")

        if "ACTION7" in action or "UNDO" in action:
            model["undo_available"] = effect.get("undo_restored", False)
            continue

        if "REPEAT" in action:
            model["action_consistent"] = effect.get("consistent", True)
            continue

        cells_changed = effect.get("cells_changed", 0)
        score_delta = effect.get("score_delta", 0)

        if cells_changed > 0:
            model["effective_actions"].append(action)
            if entry.get("data"):
                model["click_targets"].append(entry["data"])
        else:
            model["ineffective_actions"].append(action)

        if score_delta > 0:
            model["score_actions"].append((action, entry.get("data"), score_delta))

        if score_delta > best_score_delta:
            best_score_delta = score_delta
            model["best_action"] = (action, entry.get("data"))

    return model


def solve(env: ArcEnv, memory: CausalMemory, probe_log: list[dict],
          game_id: str) -> tuple[bool, int, int]:
    """Execute solve phase. Returns (won, actions_taken, level)."""
    model = analyze_probes(probe_log)
    grid, state, score, obs = env.reset()
    h, w = grid.shape
    actions_taken = 0
    level = 0

    for step in range(MAX_SOLVE_STEPS):
        # Strategy 1: If we found actions that increase score, prioritize those
        if model["score_actions"]:
            action_str, data, _ = model["score_actions"][0]
            # But also try other score actions round-robin
            if step > 0 and len(model["score_actions"]) > 1:
                idx = step % len(model["score_actions"])
                action_str, data, _ = model["score_actions"][idx]

            # For click actions, try clicking on current objects
            if "ACTION6" in action_str:
                bg = find_background_color(grid)
                objects = connected_components(grid, bg)
                if objects:
                    obj = objects[step % len(objects)]
                    cx, cy = int(obj["center"][0]), int(obj["center"][1])
                    data = {"x": cx, "y": cy}

            action = parse_action(action_str)
            grid, state, score, obs = env.step(action, data=data)
            actions_taken += 1

        # Strategy 2: Use memory retrieval
        elif memory.size > 0:
            relevant = memory.retrieve(grid, k=5)
            best_entry = None
            best_score = -float("inf")
            for entry, sim in relevant:
                sd = entry["effect"].get("score_delta", 0)
                combined = sim * 0.5 + sd * 0.5
                if combined > best_score:
                    best_score = combined
                    best_entry = entry

            if best_entry:
                action = parse_action(best_entry["action"])
                # For ACTION6, click on current objects
                data = None
                if action == GameAction.ACTION6:
                    bg = find_background_color(grid)
                    objects = connected_components(grid, bg)
                    if objects:
                        obj = objects[step % len(objects)]
                        data = {"x": int(obj['center'][0]), "y": int(obj['center'][1])}
                grid, state, score, obs = env.step(action, data=data)
                actions_taken += 1
            else:
                # Random fallback
                action = random.choice(SIMPLE_ACTIONS)
                grid, state, score, obs = env.step(action)
                actions_taken += 1

        # Strategy 3: Random from effective actions
        else:
            if model["effective_actions"]:
                action = parse_action(random.choice(model["effective_actions"]))
            else:
                action = random.choice(SIMPLE_ACTIONS)

            data = None
            if action == GameAction.ACTION6:
                bg = find_background_color(grid)
                objects = connected_components(grid, bg)
                if objects:
                    obj = random.choice(objects)
                    data = {"x": int(obj['center'][0]), "y": int(obj['center'][1])}
                else:
                    data = {"x": random.randint(0, w-1), "y": random.randint(0, h-1)}

            grid, state, score, obs = env.step(action, data=data)
            actions_taken += 1

        # Store observation in memory for future games
        if actions_taken > 1:
            delta = 0  # simplified — full delta tracking in production
            memory.store(grid, str(action), {
                "cells_changed": delta, "score_delta": 0.0,
                "state_after": state,
            }, game_id)

        if state == "WIN":
            level += 1
            return True, actions_taken, level
        if state == "GAME_OVER":
            return False, actions_taken, level

    return False, actions_taken, level
