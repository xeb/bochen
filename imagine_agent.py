"""Imagine Then Act agent — JAX-powered world model + GPU BFS.

Strategy:
  Phase 1: PROBE — ~100 API calls to learn game mechanics
  Phase 2: BUILD — Train a deterministic grid-to-grid world model on GPU
  Phase 3: SEARCH — BFS/DFS in the model's imagination (GPU, 0 API calls)
  Phase 4: EXECUTE — Play the found solution via API (~20 calls)
"""

import sys
import os
import time
import hashlib
import numpy as np
from collections import deque
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.env_wrapper import ArcEnv
from shared.perception import find_background_color, connected_components
from arcengine import GameAction


# ---------------------------------------------------------------------------
# World model: predicts next grid from current grid + action
# ---------------------------------------------------------------------------

class GridWorldModel(nn.Module):
    """Predicts next_grid from (current_grid, action).

    Input:  grid (B, 16, 64, 64) one-hot NCHW + action (B,) int
    Output: logits (B, 16, 64, 64) per-cell color probabilities
    """
    num_actions: int = 8

    @nn.compact
    def __call__(self, grid_onehot, action):
        # action embedding broadcast over spatial dims
        act_emb = nn.Embed(num_embeddings=self.num_actions, features=16)(action)  # (B, 16)
        act_map = jnp.broadcast_to(
            act_emb[:, :, None, None],
            (grid_onehot.shape[0], 16, grid_onehot.shape[2], grid_onehot.shape[3])
        )

        # Concat grid + action -> (B, 32, H, W)
        x = jnp.concatenate([grid_onehot, act_map], axis=1)

        # NCHW -> NHWC for Flax conv
        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, H, W, 32)

        # Encoder
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        # Per-cell color prediction
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)  # (B, H, W, 16)

        # NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))  # (B, 16, H, W)
        return x


# ---------------------------------------------------------------------------
# One-hot helpers
# ---------------------------------------------------------------------------

def grid_to_onehot(grid: np.ndarray, num_colors: int = 16) -> jnp.ndarray:
    """(H, W) int grid -> (1, 16, H, W) float32 one-hot."""
    oh = jax.nn.one_hot(jnp.array(grid), num_colors)  # (H, W, 16)
    return jnp.transpose(oh, (2, 0, 1))[None]  # (1, 16, H, W)


def onehot_to_grid(onehot: jnp.ndarray) -> np.ndarray:
    """(1, 16, H, W) logits -> (H, W) int grid via argmax."""
    return np.array(jnp.argmax(onehot[0], axis=0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_world_model(transitions: list[dict], num_epochs: int = 100) -> tuple:
    """Train world model from collected transitions.

    transitions: list of {"grid_before": np.ndarray, "action": int, "grid_after": np.ndarray}
    Returns: (model, params) ready for inference
    """
    n = len(transitions)
    if n == 0:
        return None, None

    # Prepare data
    grids_before = np.stack([t["grid_before"] for t in transitions])  # (N, H, W)
    actions = np.array([t["action"] for t in transitions], dtype=np.int32)  # (N,)
    grids_after = np.stack([t["grid_after"] for t in transitions])  # (N, H, W)

    model = GridWorldModel()
    rng = jax.random.key(42)

    # Init
    h, w = grids_before.shape[1], grids_before.shape[2]
    dummy_grid = jnp.zeros((1, 16, h, w))
    dummy_act = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_grid, dummy_act)

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_before, batch_after, batch_actions):
        before_oh = jax.nn.one_hot(batch_before, 16)  # (B, H, W, 16)
        before_oh = jnp.transpose(before_oh, (0, 3, 1, 2))  # (B, 16, H, W)

        def loss_fn(params):
            logits = model.apply(params, before_oh, batch_actions)  # (B, 16, H, W)
            # Cross-entropy loss per cell
            target = batch_after  # (B, H, W) int labels
            # Reshape for softmax_cross_entropy: logits (B,16,H,W) -> (B,H,W,16)
            logits_hwc = jnp.transpose(logits, (0, 2, 3, 1))
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_hwc, target)
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    batch_size = min(64, n)
    for epoch in range(num_epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(grids_before[idx]),
                jnp.array(grids_after[idx]),
                jnp.array(actions[idx]),
            )
            epoch_loss += float(loss)
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / max(1, n_batches)
            print(f"  [imagine] Epoch {epoch+1}/{num_epochs}: loss={avg:.4f}")

    # Measure accuracy on training data
    before_oh = jax.nn.one_hot(jnp.array(grids_before), 16)
    before_oh = jnp.transpose(before_oh, (0, 3, 1, 2))
    pred_logits = model.apply(params, before_oh, jnp.array(actions))
    pred_grids = jnp.argmax(pred_logits, axis=1)  # (N, H, W)
    accuracy = float(jnp.mean(pred_grids == jnp.array(grids_after)))
    print(f"  [imagine] Training accuracy: {accuracy:.4f} ({accuracy*100:.1f}% cells correct)")

    return model, params


# ---------------------------------------------------------------------------
# GPU BFS search
# ---------------------------------------------------------------------------

def bfs_in_imagination(model, params, start_grid: np.ndarray,
                       available_actions: list[int],
                       max_depth: int = 30,
                       max_states: int = 500000) -> list[int] | None:
    """BFS over predicted states using batched JAX predictions.

    Expands all actions for a frontier of states at once, using JAX
    batch inference. No Python scoring in the inner loop — just hash
    dedup and BFS expansion.

    Returns the action sequence that reaches the most-changed state.
    """
    h, w = start_grid.shape
    num_actions = len(available_actions)
    actions_arr = jnp.array(available_actions, dtype=jnp.int32)

    # JIT batched prediction: expand one grid into N next states (one per action)
    @jax.jit
    def predict_all_actions(grid_oh):
        """Given (1,16,H,W) grid, predict next state for ALL actions at once."""
        # Repeat grid for each action: (N, 16, H, W)
        grids = jnp.repeat(grid_oh, num_actions, axis=0)
        logits = model.apply(params, grids, actions_arr)
        return jnp.argmax(logits, axis=1)  # (N, H, W) int

    def grid_hash(g):
        return hashlib.md5(np.asarray(g, dtype=np.int8).tobytes()).hexdigest()[:16]

    start_hash = grid_hash(start_grid)
    visited = {start_hash}
    grid_store = {start_hash: start_grid}

    # BFS layers
    current_frontier = [(start_hash, [])]
    states_explored = 0
    best_score = -float('inf')
    best_sequence = None

    # Simple score: cells different from start (cheap numpy op)
    start_flat = start_grid.flatten()

    t0 = time.time()

    for depth in range(max_depth):
        if not current_frontier or states_explored >= max_states:
            break

        next_frontier = []

        for g_hash, seq in current_frontier:
            if states_explored >= max_states:
                break

            current_grid = grid_store[g_hash]
            grid_oh = grid_to_onehot(current_grid)

            # Predict all next states in one JAX call
            next_grids = np.array(predict_all_actions(grid_oh))  # (N, H, W)
            states_explored += num_actions

            for act_idx in range(num_actions):
                pred_grid = next_grids[act_idx]
                p_hash = grid_hash(pred_grid)

                if p_hash not in visited:
                    visited.add(p_hash)
                    grid_store[p_hash] = pred_grid
                    new_seq = seq + [available_actions[act_idx]]

                    # Cheap score: count differing cells from start
                    score = int(np.sum(pred_grid.flatten() != start_flat))
                    if score > best_score:
                        best_score = score
                        best_sequence = new_seq

                    next_frontier.append((p_hash, new_seq))

        elapsed = time.time() - t0
        rate = states_explored / max(0.001, elapsed)
        print(f"  [imagine] BFS depth={depth+1}: {states_explored} states, "
              f"{len(visited)} unique, frontier={len(next_frontier)}, "
              f"{rate:.0f}/sec, best_score={best_score}")

        current_frontier = next_frontier

    elapsed = time.time() - t0
    print(f"  [imagine] BFS complete: {states_explored} in {elapsed:.1f}s, "
          f"{len(visited)} unique, best_score={best_score}")

    return best_sequence


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ImagineAgent:
    """Imagine Then Act: probe, model, search on GPU, execute."""

    def __init__(self):
        self.model = None
        self.params = None

    def run_episode(self, game_id: str) -> dict:
        t0 = time.time()

        # Phase 1: PROBE
        print(f"  [imagine] Phase 1: Probing {game_id}...")
        env = ArcEnv(game_id, offline=False)
        transitions, available_actions_ints = self._probe(env)
        probe_time = time.time() - t0
        print(f"  [imagine] Collected {len(transitions)} transitions in {probe_time:.1f}s")

        if len(transitions) < 10:
            return {"won": False, "actions": len(transitions), "game_id": game_id,
                    "phase": "probe_failed", "reason": "too few transitions"}

        # Phase 2: BUILD MODEL
        print(f"  [imagine] Phase 2: Training world model...")
        t1 = time.time()
        self.model, self.params = train_world_model(transitions, num_epochs=50)
        train_time = time.time() - t1
        print(f"  [imagine] Model trained in {train_time:.1f}s")

        # Phase 3: SEARCH IN IMAGINATION
        print(f"  [imagine] Phase 3: BFS in imagination...")
        t2 = time.time()
        start_grid = transitions[0]["grid_before"]
        solution = bfs_in_imagination(
            self.model, self.params, start_grid,
            available_actions_ints,
            max_depth=25,
            max_states=100000,
        )
        search_time = time.time() - t2
        print(f"  [imagine] Search took {search_time:.1f}s, "
              f"solution={'found' if solution else 'none'} "
              f"({len(solution) if solution else 0} steps)")

        if not solution:
            return {"won": False, "actions": len(transitions), "game_id": game_id,
                    "phase": "search_failed", "reason": "no solution found"}

        # Phase 4: EXECUTE
        print(f"  [imagine] Phase 4: Executing {len(solution)} actions...")
        env2 = ArcEnv(game_id, offline=False)
        grid, state, score, obs = env2.reset()
        total_actions = 0

        ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4,
                      5: GameAction.ACTION5, 6: GameAction.ACTION6,
                      7: GameAction.ACTION7}

        for act_int in solution:
            action = ACTION_MAP.get(act_int, GameAction.ACTION1)
            grid, state, score, obs = env2.step(action)
            total_actions += 1

            if state == "WIN":
                total_time = time.time() - t0
                print(f"  [imagine] *** WIN after {total_actions} actions! ***")
                return {"won": True, "actions": total_actions, "game_id": game_id,
                        "level": getattr(obs, 'levels_completed', 1),
                        "phase": "execute_win",
                        "probe_actions": len(transitions),
                        "solution_length": len(solution),
                        "probe_time": probe_time,
                        "train_time": train_time,
                        "search_time": search_time,
                        "total_time": total_time}

            if state == "GAME_OVER":
                break

        total_time = time.time() - t0
        return {"won": False, "actions": total_actions + len(transitions),
                "game_id": game_id, "phase": "execute_failed",
                "solution_length": len(solution),
                "probe_time": probe_time,
                "train_time": train_time,
                "search_time": search_time,
                "total_time": total_time}

    def _probe(self, env: ArcEnv, budget: int = 100) -> tuple[list, list]:
        """Systematic probing to collect transitions."""
        grid, state, score, obs = env.reset()
        available = [int(str(a).split(".")[-1].replace("ACTION", ""))
                     for a in env.available_actions]

        ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                      3: GameAction.ACTION3, 4: GameAction.ACTION4,
                      5: GameAction.ACTION5, 6: GameAction.ACTION6,
                      7: GameAction.ACTION7}

        transitions = []
        actions_used = 0

        # Strategy 1: Try each action from initial state
        for act_int in available:
            if actions_used >= budget:
                break
            grid, state, score, obs = env.reset()
            grid_before = grid.copy()
            action = ACTION_MAP.get(act_int, GameAction.ACTION1)
            grid, state, score, obs = env.step(action)
            transitions.append({
                "grid_before": grid_before,
                "action": act_int,
                "grid_after": grid.copy(),
            })
            actions_used += 1

            if state in ("WIN", "GAME_OVER"):
                grid, state, score, obs = env.reset()

        # Strategy 2: From initial state, do action A then action B
        for a in available:
            for b in available:
                if actions_used >= budget:
                    break
                grid, state, score, obs = env.reset()
                # Do action A
                grid_before = grid.copy()
                grid, state, score, obs = env.step(ACTION_MAP.get(a))
                transitions.append({
                    "grid_before": grid_before,
                    "action": a,
                    "grid_after": grid.copy(),
                })
                actions_used += 1

                if state in ("WIN", "GAME_OVER"):
                    grid, state, score, obs = env.reset()
                    continue

                # Do action B from new state
                grid_before = grid.copy()
                grid, state, score, obs = env.step(ACTION_MAP.get(b))
                transitions.append({
                    "grid_before": grid_before,
                    "action": b,
                    "grid_after": grid.copy(),
                })
                actions_used += 1

                if state in ("WIN", "GAME_OVER"):
                    grid, state, score, obs = env.reset()

        # Strategy 3: Random sequences from various states to cover more space
        import random
        grid, state, score, obs = env.reset()
        while actions_used < budget:
            grid_before = grid.copy()
            act_int = random.choice(available)
            grid, state, score, obs = env.step(ACTION_MAP.get(act_int))
            transitions.append({
                "grid_before": grid_before,
                "action": act_int,
                "grid_after": grid.copy(),
            })
            actions_used += 1

            if state in ("WIN", "GAME_OVER"):
                grid, state, score, obs = env.reset()

        return transitions, available


if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    agent = ImagineAgent()
    result = agent.run_episode(game)
    print(f"\nFinal: {result}")
