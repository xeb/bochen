"""JAX Experiment 2: Object-Centric World Model Agent with information-gain search."""

import sys
import os
import random
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from arcengine import GameAction
from shared.env_wrapper import ArcEnv, ALL_ACTIONS
from shared.perception import one_hot_grid, connected_components, find_background_color
from jax_models.exp2_worldmodel.models import GridEncoder, TransitionPredictor
from exp2_worldmodel.config import (
    MAX_STEPS, NUM_CANDIDATE_ACTIONS, INFO_GAIN_ALPHA_START, INFO_GAIN_ALPHA_DECAY
)


class WorldModelAgent:
    """Agent that uses a trained JAX world model for information-gain search."""

    def __init__(self, checkpoint_path: str = None):
        self.encoder = GridEncoder()
        self.predictor = TransitionPredictor(state_dim=128, action_dim=8)
        self.trained = False

        # Init params with dummy inputs
        rng = jax.random.key(0)
        r1, r2 = jax.random.split(rng)
        self.params = {
            "encoder": self.encoder.init(r1, jnp.zeros((1, 16, 64, 64))),
            "predictor": self.predictor.init(
                r2, jnp.zeros((1, 128)), jnp.zeros((1,), dtype=jnp.int32)
            ),
        }

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.action_effect_counts = {}

    def _load_checkpoint(self, path: str):
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(path)
        self.params["encoder"] = ckpt["encoder"]
        self.params["predictor"] = ckpt["predictor"]
        self.trained = True
        print(f"  [exp2/jax] Loaded world model from {path}")

    def run_episode(self, game_id: str) -> dict:
        """Run one episode using world model for action selection."""
        env = ArcEnv(game_id, offline=False, render=False)
        grid, state, score, obs = env.reset()
        h, w = grid.shape

        alpha = INFO_GAIN_ALPHA_START
        total_actions = 0
        level = 0
        prev_grid = None

        for step in range(MAX_STEPS):
            if self.trained:
                action, data = self._select_action_model(grid, h, w, alpha)
            else:
                action, data = self._select_action_heuristic(grid, h, w, step)

            prev_grid = grid.copy()
            grid, state, new_score, obs = env.step(action, data=data)
            total_actions += 1

            delta = int(np.sum(grid != prev_grid))
            action_idx = ALL_ACTIONS.index(action) if action in ALL_ACTIONS else 0
            self.action_effect_counts.setdefault(action_idx, {})
            effect_key = "changed" if delta > 0 else "no_change"
            self.action_effect_counts[action_idx][effect_key] = \
                self.action_effect_counts[action_idx].get(effect_key, 0) + 1

            alpha = max(0.0, alpha - INFO_GAIN_ALPHA_DECAY)
            score = new_score

            if state == "WIN":
                level += 1
                return {"won": True, "level": level, "actions": total_actions,
                        "game_id": game_id, "trained": self.trained}
            if state == "GAME_OVER":
                return {"won": False, "level": level, "actions": total_actions,
                        "game_id": game_id, "trained": self.trained}

        return {"won": False, "level": level, "actions": total_actions,
                "game_id": game_id, "trained": self.trained}

    def _select_action_model(self, grid, h, w, alpha):
        """Use world model to score candidate actions by information gain."""
        x = jnp.array(one_hot_grid(grid))[None, ...]  # (1, 16, 64, 64)

        state_enc = self.encoder.apply(self.params["encoder"], x)  # (1, 128)

        # Generate candidate actions
        candidates = []
        for i in range(7):
            candidates.append((i, None))
        bg = find_background_color(grid)
        objects = connected_components(grid, bg)
        for obj in objects[:20]:
            cx, cy = int(obj["center"][0]), int(obj["center"][1])
            candidates.append((5, {"x": cx, "y": cy}))
        while len(candidates) < NUM_CANDIDATE_ACTIONS:
            rx, ry = random.randint(0, w - 1), random.randint(0, h - 1)
            candidates.append((5, {"x": rx, "y": ry}))

        # Batch predict
        n = len(candidates)
        state_batch = jnp.broadcast_to(state_enc, (n, 128))
        action_batch = jnp.array([c[0] for c in candidates], dtype=jnp.int32)
        preds = self.predictor.apply(self.params["predictor"], state_batch, action_batch)

        pred_states = preds["next_state"]  # (n, 128)
        mean_pred = pred_states.mean(axis=0, keepdims=True)
        variance = ((pred_states - mean_pred) ** 2).sum(axis=-1)
        state_change = ((pred_states - state_batch) ** 2).sum(axis=-1)
        scores = alpha * variance + (1 - alpha) * state_change

        best_idx = int(jnp.argmax(scores))

        action_idx, data = candidates[best_idx]
        action = ALL_ACTIONS[action_idx]
        return action, data

    def _select_action_heuristic(self, grid, h, w, step):
        """Fallback: systematic exploration without trained model."""
        if step < 7:
            return ALL_ACTIONS[step % 7], None

        bg = find_background_color(grid)
        objects = connected_components(grid, bg)
        if objects and step - 7 < len(objects):
            obj = objects[(step - 7) % len(objects)]
            cx, cy = int(obj["center"][0]), int(obj["center"][1])
            return GameAction.ACTION6, {"x": cx, "y": cy}

        if random.random() < 0.3:
            return random.choice(ALL_ACTIONS[:5]), None
        else:
            rx, ry = random.randint(0, w - 1), random.randint(0, h - 1)
            return GameAction.ACTION6, {"x": rx, "y": ry}
