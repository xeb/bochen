#!/usr/bin/env python3
"""Bochen Three-Phase Agent for ARC-AGI-3.

Phase 1: EXPLORE — try every control from multiple states, record transitions,
          move objects off cells to reveal hidden targets.
Phase 2: PERCEIVE — CNN trained on exploration data classifies every cell as
          background | wall | movable | target. Trained across ALL games.
Phase 3: SOLVE — A* in the world model using Manhattan(movable→target) heuristic.
          If A* fails, escalate to LLM (Claude) for goal reasoning.
"""

import sys
import os
import time
import heapq
import hashlib
import random
import json
import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.env_wrapper import ArcEnv
from arcengine import GameAction

ACTION_MAP = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2,
    3: GameAction.ACTION3, 4: GameAction.ACTION4,
    5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


# ============================================================================
# MODELS (all JAX/Flax)
# ============================================================================

class GridWorldModel(nn.Module):
    """Predicts next grid from (current_grid, action). Deterministic dynamics."""
    num_actions: int = 8

    @nn.compact
    def __call__(self, grid_onehot, action):
        act_emb = nn.Embed(num_embeddings=self.num_actions, features=16)(action)
        act_map = jnp.broadcast_to(
            act_emb[:, :, None, None],
            (grid_onehot.shape[0], 16, grid_onehot.shape[2], grid_onehot.shape[3])
        )
        x = jnp.concatenate([grid_onehot, act_map], axis=1)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(1, 1))(x)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return x


class CellRoleCNN(nn.Module):
    """Classifies each cell into a role: 0=background, 1=wall, 2=movable, 3=target.

    Input:  (B, C, H, W) where C = 16 (one-hot color) + 4 (movement stats)
    Output: (B, 4, H, W) per-cell role logits
    """
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=4, kernel_size=(1, 1))(x)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW
        return x


# ============================================================================
# HELPERS
# ============================================================================

def grid_to_onehot(grid: np.ndarray, num_colors: int = 16) -> jnp.ndarray:
    oh = jax.nn.one_hot(jnp.array(grid), num_colors)
    return jnp.transpose(oh, (2, 0, 1))[None]  # (1, 16, H, W)


def grid_hash(g: np.ndarray) -> str:
    return hashlib.md5(np.asarray(g, dtype=np.int8).tobytes()).hexdigest()[:16]


def get_available_ints(env: ArcEnv) -> list[int]:
    return [int(str(a).split(".")[-1].replace("ACTION", ""))
            for a in env.available_actions]


# ============================================================================
# PHASE 1: EXPLORE
# ============================================================================

def explore(game_id: str, budget: int = 120) -> dict:
    """Systematically explore a game. Returns exploration results."""
    env = ArcEnv(game_id, offline=False)
    grid0, state, score, obs = env.reset()
    available = get_available_ints(env)
    h, w = grid0.shape

    transitions = []
    actions_used = 0

    # Track per-cell movement: how many times each cell changed value
    move_counts = np.zeros((h, w), dtype=np.int32)
    # Track which cells were ever occupied by non-background after movement
    ever_occupied = np.zeros((h, w), dtype=bool)
    # Track initial grid for reference
    initial_grid = grid0.copy()

    def record(env_inst, act_int):
        nonlocal actions_used
        g_before = env_inst._last_obs
        grid_before, _, _, _ = env_inst._extract()
        grid_before = grid_before.copy()
        grid_after, st, sc, _ = env_inst.step(ACTION_MAP[act_int])
        grid_after = grid_after.copy()
        transitions.append({
            "grid_before": grid_before,
            "action": act_int,
            "grid_after": grid_after,
        })
        diff = grid_before != grid_after
        move_counts[diff] += 1
        ever_occupied[grid_after != initial_grid[0, 0]] = True  # rough heuristic
        actions_used += 1
        return grid_after, st

    # --- Strategy 1: Each action from initial state ---
    for act in available:
        if actions_used >= budget:
            break
        env = ArcEnv(game_id, offline=False)
        env.reset()
        _, st = record(env, act)

    # --- Strategy 2: Two-step combos (learn state-dependent effects) ---
    for a in available:
        for b in available:
            if actions_used >= budget - 1:
                break
            env = ArcEnv(game_id, offline=False)
            env.reset()
            _, st = record(env, a)
            if st in ("WIN", "GAME_OVER"):
                continue
            _, st = record(env, b)

    # --- Strategy 3: Push objects maximally in each direction to reveal targets ---
    # Do each action 8 times to push objects far from start
    pushed_grids = []
    for act in available:
        if actions_used >= budget - 8:
            break
        env = ArcEnv(game_id, offline=False)
        env.reset()
        for _ in range(8):
            if actions_used >= budget:
                break
            _, st = record(env, act)
            if st in ("WIN", "GAME_OVER"):
                break
        pushed_grids.append(env._extract()[0].copy())

    # --- Strategy 4: Random play to fill remaining budget ---
    if actions_used < budget:
        env = ArcEnv(game_id, offline=False)
        env.reset()
        while actions_used < budget:
            act = random.choice(available)
            _, st = record(env, act)
            if st in ("WIN", "GAME_OVER"):
                env = ArcEnv(game_id, offline=False)
                env.reset()

    print(f"  [explore] {game_id}: {len(transitions)} transitions, "
          f"{len(available)} actions, grid={h}x{w}")

    return {
        "game_id": game_id,
        "transitions": transitions,
        "available_actions": available,
        "initial_grid": initial_grid,
        "pushed_grids": pushed_grids,
        "move_counts": move_counts,
        "grid_shape": (h, w),
    }


# ============================================================================
# PHASE 2: PERCEIVE (Cell Role CNN)
# ============================================================================

def build_role_labels(exploration: dict) -> np.ndarray:
    """Generate per-cell role labels from exploration data.

    Returns (H, W) int array: 0=background, 1=wall, 2=movable, 3=target

    Strategy: compare initial grid to maximally-pushed grids.
    - Cells that are IDENTICAL across initial AND all pushed grids = truly static
    - Among static: most common color = background, next = wall
    - Cells that DIFFER between initial and any pushed grid = contain objects
    - In the INITIAL grid, the object cells are labeled movable (2)
    - In pushed grids, cells that NEWLY appeared with non-bg non-wall color
      at positions where objects LEFT = revealed targets (3)
    """
    initial = exploration["initial_grid"]
    pushed_grids = exploration.get("pushed_grids", [])
    h, w = exploration["grid_shape"]

    # Step 1: Find ALL colors in initial grid, sorted by frequency
    color_counts = np.bincount(initial.flatten(), minlength=16)
    colors_by_freq = np.argsort(-color_counts)  # most common first
    bg_color = colors_by_freq[0]

    # Step 2: Find cells that moved (differ between initial and ANY pushed grid)
    # Then expand: if ANY cell of a color moved, ALL cells of that color are movable
    raw_moved = np.zeros((h, w), dtype=bool)
    for pg in pushed_grids:
        raw_moved |= (initial != pg)

    # Find colors that have at least one moved cell
    moved_colors = set()
    for c in range(16):
        color_mask = (initial == c)
        if np.any(raw_moved & color_mask):
            moved_colors.add(c)
    # Don't mark the two most common colors (bg/wall) as movable even if
    # objects slide over them causing transient changes
    moved_colors.discard(int(bg_color))
    if len(moved_colors) > 4:
        # Too many — only keep the rarer ones
        # Sort by frequency, keep the rarest half
        mc_list = sorted(moved_colors, key=lambda c: color_counts[c])
        moved_colors = set(mc_list[:len(mc_list)//2 + 1])

    # Movable mask: all cells whose color is in the moved set
    moved_mask = np.zeros((h, w), dtype=bool)
    for c in moved_colors:
        moved_mask |= (initial == c)

    # Step 3: Find what's UNDER moved objects (revealed in pushed grids)
    # Collect all colors revealed at moved-from positions
    revealed_at_moved = {}  # (y,x) -> set of revealed colors
    for pg in pushed_grids:
        diff_mask = (initial != pg)
        for y, x in zip(*np.where(diff_mask)):
            revealed_at_moved.setdefault((y, x), set()).add(int(pg[y, x]))

    # Step 4: Classify cells
    # - bg_color cells that never moved = background (0)
    # - Second most common static color = wall (1)
    # - Cells that moved = movable (2) in initial grid
    # - Rare colors that are STATIC or revealed under objects = target (3)
    static_mask = ~moved_mask
    static_non_bg = static_mask & (initial != bg_color)

    # Wall = most common static non-bg color
    static_non_bg_colors = initial[static_non_bg]
    if len(static_non_bg_colors) > 0:
        u, c = np.unique(static_non_bg_colors, return_counts=True)
        wall_color = u[np.argmax(c)]
    else:
        wall_color = bg_color

    labels = np.zeros((h, w), dtype=np.int32)
    labels[initial == bg_color] = 0  # background
    labels[static_mask & (initial == wall_color)] = 1  # wall
    labels[moved_mask] = 2  # movable

    # Target candidates: RARE static cells that are not bg, wall, or movable colors
    # In Sokoban-like games, targets are small markers (few cells), not large areas
    static_non_bg_wall = static_mask & (initial != bg_color) & (initial != wall_color)

    # Among static non-bg non-wall cells, find colors that are RARE
    # (targets should be fewer cells than movable objects)
    n_movable = moved_mask.sum()
    candidate_colors = {}
    for c in range(16):
        if c == bg_color or c == wall_color or c in moved_colors:
            continue
        count = int(np.sum(static_non_bg_wall & (initial == c)))
        if count > 0:
            candidate_colors[c] = count

    # Label as target: rare colors (each color <= movable count)
    # Also include colors revealed under objects
    target_colors = set()
    for c, count in candidate_colors.items():
        if count <= max(n_movable, 50):  # targets shouldn't be bigger than objects
            target_colors.add(c)

    target_mask = np.zeros((h, w), dtype=bool)
    for c in target_colors:
        target_mask |= (static_non_bg_wall & (initial == c))

    # Also: cells revealed under objects that have rare non-bg/wall colors
    for (y, x), revealed_colors in revealed_at_moved.items():
        for rc in revealed_colors:
            if rc != bg_color and rc != wall_color and rc not in moved_colors:
                target_mask[y, x] = True

    labels[target_mask] = 3

    n_bg = np.sum(labels == 0)
    n_wall = np.sum(labels == 1)
    n_mov = np.sum(labels == 2)
    n_tgt = np.sum(labels == 3)
    print(f"  [perceive] Labels: bg={n_bg} wall={n_wall} movable={n_mov} target={n_tgt}")
    print(f"  [perceive] Movable colors: {moved_colors}, Target colors: {target_colors}")

    return labels, moved_colors, target_colors


def build_perception_features(grid: np.ndarray, move_counts: np.ndarray) -> jnp.ndarray:
    """Build CNN input: 16-channel one-hot color + 4 channels of movement stats.

    Returns (1, 20, H, W) float32.
    """
    h, w = grid.shape
    # One-hot color
    oh = np.zeros((16, h, w), dtype=np.float32)
    for c in range(16):
        oh[c] = (grid == c).astype(np.float32)

    # Movement stats channels
    mc = move_counts.astype(np.float32)
    mc_norm = mc / max(mc.max(), 1.0)
    moved_binary = (mc > 0).astype(np.float32)
    never_moved = (mc == 0).astype(np.float32)
    high_movement = (mc > np.median(mc[mc > 0]) if mc.max() > 0 else mc).astype(np.float32)

    features = np.concatenate([
        oh,                            # 16 channels: color
        mc_norm[None],                 # 1 channel: normalized move count
        moved_binary[None],            # 1 channel: ever moved?
        never_moved[None],             # 1 channel: never moved?
        high_movement[None],           # 1 channel: high movement?
    ], axis=0)  # (20, H, W)

    return jnp.array(features[None])  # (1, 20, H, W)


def train_role_cnn(explorations: list[dict], num_epochs: int = 100) -> tuple:
    """Train CellRoleCNN on labeled data from ALL games' exploration.

    Returns (model, params).
    """
    all_features = []
    all_labels = []

    for exp in explorations:
        labels, _, _ = build_role_labels(exp)
        initial = exp["initial_grid"]
        mc = exp["move_counts"]

        # Generate training samples from multiple grids in the exploration
        seen_grids = {grid_hash(initial)}
        grids_to_label = [initial]

        for t in exp["transitions"]:
            for g in [t["grid_before"], t["grid_after"]]:
                h = grid_hash(g)
                if h not in seen_grids:
                    seen_grids.add(h)
                    grids_to_label.append(g)
                if len(grids_to_label) >= 50:  # cap per game
                    break

        for g in grids_to_label:
            feat = build_perception_features(g, mc)
            all_features.append(np.array(feat[0]))  # (20, H, W)
            # Labels are based on initial grid roles — transfer to current grid
            # Movable cells in current grid are where movable-colored cells ARE now
            all_labels.append(labels)

    X = jnp.array(np.stack(all_features))  # (N, 20, H, W)
    Y = jnp.array(np.stack(all_labels))    # (N, H, W) int

    n = len(X)
    print(f"  [perceive] Training CNN on {n} samples from {len(explorations)} games")

    model = CellRoleCNN()
    rng = jax.random.key(99)
    h, w = Y.shape[1], Y.shape[2]
    params = model.init(rng, jnp.zeros((1, 20, h, w)))

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(params):
            logits = model.apply(params, batch_x)  # (B, 4, H, W)
            logits_hwc = jnp.transpose(logits, (0, 2, 3, 1))  # (B, H, W, 4)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_hwc, batch_y)
            return jnp.mean(loss)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    batch_size = min(32, n)
    for epoch in range(num_epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            params, opt_state, loss = train_step(params, opt_state, X[idx], Y[idx])
            epoch_loss += float(loss)
            nb += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  [perceive] Epoch {epoch+1}/{num_epochs}: loss={epoch_loss/nb:.4f}")

    # Measure accuracy
    pred = jnp.argmax(model.apply(params, X), axis=1)  # (N, H, W)
    acc = float(jnp.mean(pred == Y))
    print(f"  [perceive] CNN accuracy: {acc:.4f} ({acc*100:.1f}%)")

    return model, params


def perceive(role_model, role_params, grid: np.ndarray,
             move_counts: np.ndarray) -> dict:
    """Run perception on a single grid. Returns role map and extracted positions."""
    feat = build_perception_features(grid, move_counts)
    logits = role_model.apply(role_params, feat)  # (1, 4, H, W)
    roles = np.array(jnp.argmax(logits[0], axis=0))  # (H, W) int

    # Extract positions
    movable_positions = list(zip(*np.where(roles == 2)))  # [(y,x), ...]
    target_positions = list(zip(*np.where(roles == 3)))

    return {
        "roles": roles,
        "movable": movable_positions,
        "targets": target_positions,
        "n_movable": len(movable_positions),
        "n_targets": len(target_positions),
    }


# ============================================================================
# PHASE 3: SOLVE (A* with Manhattan heuristic, escalate to LLM)
# ============================================================================

def manhattan_heuristic(grid: np.ndarray, movable_colors: set,
                        target_positions_arr: np.ndarray) -> float:
    """Sum of min Manhattan distances from movable-colored cells to targets.

    Uses color-based detection (fast) instead of CNN inference.
    target_positions_arr: (T, 2) numpy array of (y, x) target positions.
    """
    if len(target_positions_arr) == 0:
        return 0.0

    # Find movable cells by color (O(1) per color)
    movable_yx = []
    for c in movable_colors:
        positions = np.argwhere(grid == c)
        movable_yx.append(positions)

    if not movable_yx:
        return 0.0

    movable_yx = np.concatenate(movable_yx)
    if len(movable_yx) == 0:
        return 0.0

    # Vectorized Manhattan distance computation
    # movable_yx: (M, 2), targets: (T, 2)
    # Compute pairwise distances: (M, T)
    diffs = np.abs(movable_yx[:, None, :] - target_positions_arr[None, :, :])
    dists = diffs.sum(axis=2)  # (M, T)
    return float(dists.min(axis=1).sum())


def astar_solve(world_model, world_params, role_model, role_params,
                start_grid: np.ndarray, move_counts: np.ndarray,
                available_actions: list[int], movable_colors: set,
                max_states: int = 500000, max_depth: int = 80) -> list[int] | None:
    """A* search using Manhattan(movable→target) as heuristic.

    Uses color-based movable detection (fast) instead of CNN per state.
    Target positions are fixed (from initial perception).
    """
    actions_arr = jnp.array(available_actions, dtype=jnp.int32)
    num_actions = len(available_actions)

    @jax.jit
    def predict_all_actions(grid_oh):
        grids = jnp.repeat(grid_oh, num_actions, axis=0)
        logits = world_model.apply(world_params, grids, actions_arr)
        return jnp.argmax(logits, axis=1)

    # Get targets from initial perception (fixed for the whole search)
    perception = perceive(role_model, role_params, start_grid, move_counts)
    targets = perception["targets"]

    if not targets:
        print("  [solve] No targets detected — A* cannot run without goal")
        return None

    target_arr = np.array(targets)  # (T, 2) for fast heuristic
    n_mov = perception["n_movable"]
    n_tgt = len(targets)
    print(f"  [solve] A* search: {n_mov} movable cells ({movable_colors}), {n_tgt} targets")

    start_h_val = manhattan_heuristic(start_grid, movable_colors, target_arr)
    s_hash = grid_hash(start_grid)

    open_set = [(start_h_val, 0, s_hash, [])]
    visited = {s_hash}
    grid_store = {s_hash: start_grid}
    states_explored = 0
    best_h = start_h_val

    t0 = time.time()

    while open_set and states_explored < max_states:
        f, g, g_hash, seq = heapq.heappop(open_set)

        if g >= max_depth:
            continue

        current_grid = grid_store[g_hash]
        grid_oh = grid_to_onehot(current_grid)
        next_grids = np.array(predict_all_actions(grid_oh))
        states_explored += num_actions

        for act_idx in range(num_actions):
            pred_grid = next_grids[act_idx]
            p_hash = grid_hash(pred_grid)

            if p_hash in visited:
                continue
            visited.add(p_hash)
            grid_store[p_hash] = pred_grid

            new_seq = seq + [available_actions[act_idx]]

            # Fast heuristic: color-based movable detection, no CNN
            h_cost = manhattan_heuristic(pred_grid, movable_colors, target_arr)

            if h_cost < best_h:
                best_h = h_cost
                elapsed = time.time() - t0
                print(f"  [solve] A* depth={g+1}: h={h_cost:.0f} (best), "
                      f"{states_explored} explored, {elapsed:.1f}s")

            if h_cost == 0:
                elapsed = time.time() - t0
                print(f"  [solve] *** SOLUTION: {len(new_seq)} steps, "
                      f"{states_explored} explored, {elapsed:.1f}s ***")
                return new_seq

            new_g = g + 1
            new_f = new_g + h_cost
            heapq.heappush(open_set, (new_f, new_g, p_hash, new_seq))

        if states_explored % 10000 == 0:
            elapsed = time.time() - t0
            rate = states_explored / max(0.001, elapsed)
            print(f"  [solve] A*: {states_explored} states, "
                  f"{len(visited)} unique, best_h={best_h:.0f}, {rate:.0f}/sec")

    elapsed = time.time() - t0
    print(f"  [solve] A* exhausted: {states_explored} in {elapsed:.1f}s, best_h={best_h:.0f}")
    return None


def escalate_to_llm(exploration: dict, perception_result: dict) -> list[int] | None:
    """When A* fails, ask Claude to reason about the goal."""
    try:
        import anthropic
    except ImportError:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  [escalate] No anthropic SDK and no API key — cannot escalate")
            return None
        # Use raw requests
        import anthropic

    game_id = exploration["game_id"]
    roles = perception_result["roles"]
    n_mov = perception_result["n_movable"]
    n_tgt = perception_result["n_targets"]
    available = exploration["available_actions"]

    # Build a text description
    from shared.perception import grid_to_ascii
    initial = exploration["initial_grid"]
    ascii_grid = grid_to_ascii(initial)

    # Compact role map
    role_chars = {0: '.', 1: '#', 2: 'M', 3: 'T'}
    role_ascii = '\n'.join(
        ''.join(role_chars.get(int(roles[y, x]), '?') for x in range(roles.shape[1]))
        for y in range(roles.shape[0])
    )

    prompt = f"""I'm solving an ARC-AGI-3 puzzle (game: {game_id}).

Available actions: {available} (likely directional movement)

Grid (hex colors):
{ascii_grid[:2000]}

Role map (. = background, # = wall, M = movable, T = target):
{role_ascii[:2000]}

I found {n_mov} movable cells and {n_tgt} target cells.
A* search with Manhattan distance heuristic didn't find a solution.

What is likely the win condition? What strategy should I try?
Respond with a JSON action sequence to try, e.g. {{"actions": [1, 3, 3, 2, 4]}}"""

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        print(f"  [escalate] LLM response: {text[:200]}")

        # Try to parse action sequence
        import re
        match = re.search(r'"actions"\s*:\s*\[([^\]]+)\]', text)
        if match:
            actions = [int(x.strip()) for x in match.group(1).split(",")]
            return actions
    except Exception as e:
        print(f"  [escalate] LLM call failed: {e}")

    return None


# ============================================================================
# MAIN AGENT
# ============================================================================

class ThreePhaseAgent:
    """Three-phase agent: Explore → Perceive → Solve."""

    def __init__(self):
        self.world_model = None
        self.world_params = None
        self.role_model = None
        self.role_params = None
        self.all_explorations = []

    def run_episode(self, game_id: str) -> dict:
        t0 = time.time()

        # ── Phase 1: EXPLORE ──
        print(f"\n  === Phase 1: EXPLORE ({game_id}) ===")
        exploration = explore(game_id, budget=120)
        self.all_explorations.append(exploration)
        transitions = exploration["transitions"]
        available = exploration["available_actions"]
        explore_time = time.time() - t0
        print(f"  [explore] Done in {explore_time:.1f}s")

        # ── Train world model from ALL explorations ──
        print(f"\n  === Training World Model ({len(self.all_explorations)} games) ===")
        t1 = time.time()
        all_transitions = []
        for exp in self.all_explorations:
            all_transitions.extend(exp["transitions"])
        self.world_model, self.world_params = self._train_world_model(all_transitions)
        wm_time = time.time() - t1

        # ── Phase 2: PERCEIVE ──
        print(f"\n  === Phase 2: PERCEIVE ({len(self.all_explorations)} games) ===")
        t2 = time.time()
        # Get movable colors from label builder for fast A* heuristic
        _, movable_colors, target_colors = build_role_labels(exploration)

        self.role_model, self.role_params = train_role_cnn(
            self.all_explorations, num_epochs=80
        )
        perception = perceive(
            self.role_model, self.role_params,
            exploration["initial_grid"], exploration["move_counts"]
        )
        perceive_time = time.time() - t2
        print(f"  [perceive] Done in {perceive_time:.1f}s: "
              f"{perception['n_movable']} movable, {perception['n_targets']} targets")

        # ── Phase 3: SOLVE ──
        print(f"\n  === Phase 3: SOLVE ({game_id}) ===")
        t3 = time.time()

        solution = astar_solve(
            self.world_model, self.world_params,
            self.role_model, self.role_params,
            exploration["initial_grid"], exploration["move_counts"],
            available, movable_colors,
            max_states=500000, max_depth=80,
        )

        if solution is None:
            print("  [solve] A* failed — escalating to LLM...")
            solution = escalate_to_llm(exploration, perception)

        solve_time = time.time() - t3

        if solution is None:
            total_time = time.time() - t0
            return {"won": False, "actions": len(transitions),
                    "game_id": game_id, "phase": "solve_failed",
                    "explore_time": explore_time, "perceive_time": perceive_time,
                    "solve_time": solve_time, "total_time": total_time,
                    "n_movable": perception["n_movable"],
                    "n_targets": perception["n_targets"]}

        # ── EXECUTE ──
        print(f"\n  === EXECUTE: {len(solution)} actions on {game_id} ===")
        env = ArcEnv(game_id, offline=False)
        grid, state, score, obs = env.reset()
        total_actions = 0

        for act_int in solution:
            action = ACTION_MAP.get(act_int, GameAction.ACTION1)
            grid, state, score, obs = env.step(action)
            total_actions += 1

            if state == "WIN":
                total_time = time.time() - t0
                level = getattr(obs, 'levels_completed', 1)
                print(f"  *** WIN on {game_id} level {level} "
                      f"after {total_actions} actions in {total_time:.1f}s! ***")
                return {"won": True, "actions": total_actions,
                        "game_id": game_id, "level": level,
                        "phase": "execute_win",
                        "solution_length": len(solution),
                        "explore_time": explore_time,
                        "perceive_time": perceive_time,
                        "solve_time": solve_time,
                        "total_time": total_time}

            if state == "GAME_OVER":
                break

        total_time = time.time() - t0
        return {"won": False, "actions": total_actions + len(transitions),
                "game_id": game_id, "phase": "execute_failed",
                "solution_length": len(solution),
                "explore_time": explore_time, "perceive_time": perceive_time,
                "solve_time": solve_time, "total_time": total_time}

    def _train_world_model(self, transitions, num_epochs=50):
        n = len(transitions)
        grids_before = np.stack([t["grid_before"] for t in transitions])
        actions = np.array([t["action"] for t in transitions], dtype=np.int32)
        grids_after = np.stack([t["grid_after"] for t in transitions])

        model = GridWorldModel()
        rng = jax.random.key(42)
        h, w = grids_before.shape[1], grids_before.shape[2]
        params = model.init(rng, jnp.zeros((1, 16, h, w)), jnp.zeros((1,), dtype=jnp.int32))
        tx = optax.adam(1e-3)
        opt_state = tx.init(params)

        @jax.jit
        def train_step(params, opt_state, b_before, b_after, b_actions):
            boh = jax.nn.one_hot(b_before, 16)
            boh = jnp.transpose(boh, (0, 3, 1, 2))
            def loss_fn(p):
                logits = model.apply(p, boh, b_actions)
                logits_hwc = jnp.transpose(logits, (0, 2, 3, 1))
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_hwc, b_after))
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt, loss

        bs = min(64, n)
        for epoch in range(num_epochs):
            perm = np.random.permutation(n)
            eloss, nb = 0.0, 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                params, opt_state, loss = train_step(
                    params, opt_state,
                    jnp.array(grids_before[idx]),
                    jnp.array(grids_after[idx]),
                    jnp.array(actions[idx]),
                )
                eloss += float(loss)
                nb += 1
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"  [world model] Epoch {epoch+1}/{num_epochs}: loss={eloss/nb:.4f}")

        boh = jax.nn.one_hot(jnp.array(grids_before), 16)
        boh = jnp.transpose(boh, (0, 3, 1, 2))
        pred = jnp.argmax(model.apply(params, boh, jnp.array(actions)), axis=1)
        acc = float(jnp.mean(pred == jnp.array(grids_after)))
        print(f"  [world model] Accuracy: {acc*100:.1f}%")
        return model, params


if __name__ == "__main__":
    games = sys.argv[1:] if len(sys.argv) > 1 else ["ls20", "ft09", "vc33"]
    agent = ThreePhaseAgent()
    for game_id in games:
        result = agent.run_episode(game_id)
        print(f"\n{'='*60}")
        print(f"  RESULT {game_id}: {'WIN' if result['won'] else 'LOST'}")
        print(f"  {json.dumps({k:v for k,v in result.items() if k != 'transitions'}, indent=2)}")
        print(f"{'='*60}")
