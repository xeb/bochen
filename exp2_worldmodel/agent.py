"""Experiment 2: Object-Centric World Model Agent with information-gain search."""

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcengine import GameAction
from shared.env_wrapper import ArcEnv, ALL_ACTIONS
from shared.perception import one_hot_grid, connected_components, find_background_color
from exp2_worldmodel.models import GridEncoder, TransitionPredictor
from exp2_worldmodel.config import (
    MAX_STEPS, NUM_CANDIDATE_ACTIONS, INFO_GAIN_ALPHA_START, INFO_GAIN_ALPHA_DECAY
)


class WorldModelAgent:
    """Agent that uses a trained world model for information-gain search."""

    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = device
        self.encoder = GridEncoder().to(device)
        self.predictor = TransitionPredictor(state_dim=128, action_dim=8).to(device)
        self.trained = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        # Belief over action effects: track what each action does
        self.action_effect_counts = {}  # action_idx -> {effect_type -> count}

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.trained = True
        print(f"  [exp2] Loaded world model from {path}")

    def run_episode(self, game_id: str) -> dict:
        """Run one episode using world model for action selection."""
        env = ArcEnv(game_id, offline=False, render=False)
        grid, state, score, obs = env.reset()
        h, w = grid.shape

        self.encoder.eval()
        self.predictor.eval()
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

            # Track action effects
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
        x = torch.from_numpy(one_hot_grid(grid)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            state_enc = self.encoder(x)  # (1, 128)

            # Generate candidate actions
            candidates = []
            # All simple actions
            for i in range(7):
                candidates.append((i, None))
            # Sample ACTION6 coordinates on object centroids
            bg = find_background_color(grid)
            objects = connected_components(grid, bg)
            for obj in objects[:20]:
                cx, cy = int(obj["center"][0]), int(obj["center"][1])
                candidates.append((5, {"x": cx, "y": cy}))  # ACTION6 = index 5
            # Random positions if not enough
            while len(candidates) < NUM_CANDIDATE_ACTIONS:
                rx, ry = random.randint(0, w - 1), random.randint(0, h - 1)
                candidates.append((5, {"x": rx, "y": ry}))

            # Batch predict
            n = len(candidates)
            state_batch = state_enc.expand(n, -1)
            action_batch = torch.tensor(
                [c[0] for c in candidates], dtype=torch.long, device=self.device
            )
            preds = self.predictor(state_batch, action_batch)

            # Information gain: variance of predictions = how uncertain we are
            # Higher variance in predicted next state = more to learn
            pred_states = preds["next_state"]  # (n, 128)
            mean_pred = pred_states.mean(dim=0, keepdim=True)
            variance = ((pred_states - mean_pred) ** 2).sum(dim=-1)  # (n,)

            # Goal progress: magnitude of state change
            state_change = ((pred_states - state_batch) ** 2).sum(dim=-1)  # (n,)

            # Combined score
            scores = alpha * variance + (1 - alpha) * state_change

            best_idx = scores.argmax().item()

        action_idx, data = candidates[best_idx]
        action = ALL_ACTIONS[action_idx]
        return action, data

    def _select_action_heuristic(self, grid, h, w, step):
        """Fallback: systematic exploration without trained model."""
        # Cycle through simple actions first
        if step < 7:
            return ALL_ACTIONS[step % 7], None

        # Then click on objects
        bg = find_background_color(grid)
        objects = connected_components(grid, bg)
        if objects and step - 7 < len(objects):
            obj = objects[(step - 7) % len(objects)]
            cx, cy = int(obj["center"][0]), int(obj["center"][1])
            return GameAction.ACTION6, {"x": cx, "y": cy}

        # Random
        if random.random() < 0.3:
            return random.choice(ALL_ACTIONS[:5]), None
        else:
            rx, ry = random.randint(0, w - 1), random.randint(0, h - 1)
            return GameAction.ACTION6, {"x": rx, "y": ry}
