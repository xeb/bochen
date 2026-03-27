# 10 GPU-Accelerated ARC-AGI-3 Experiments

**Hardware**: RTX 4090 (24GB VRAM), PyTorch 2.8.0+cu128, CUDA 12.4
**Goal**: Fast failure — each experiment should produce a measurable RHAE signal within 1-4 hours. Kill anything that doesn't beat random by experiment end.

---

## 1. Vision Transformer State Encoder + PPO

**Idea**: Treat each game frame as a 64x64x16 one-hot image. Train a small ViT to encode states, feed into PPO policy head. The GPU handles both forward passes and rollout batching at ~2000 FPS local mode.

**Why it might work**: Grid states are spatial — convolutions/attention over patches could discover spatial invariants that generalize across levels.

**Fast-fail signal**: If PPO reward curve is flat after 50K steps on a single game, kill it.

**Implementation**:
```bash
uv add --script='exp01_vit_ppo.py' 'torch' 'gymnasium' 'arc-agi' 'einops'
```

```python
# exp01_vit_ppo.py
# 1. Wrap arc_agi env in a gymnasium-compatible wrapper
# 2. Observation: env frame grid -> one-hot encode 16 colors -> (16, H, W) tensor on GPU
# 3. ViT encoder: patch_size=4, dim=128, depth=4, heads=4 (~2M params, fits easily)
# 4. Policy head: action_type (7 discrete) + x,y coords (continuous, discretized to grid)
# 5. PPO with GAE, batch 2048 steps, 4 epochs per update
# 6. Run OFFLINE mode at max FPS, log episode reward every 100 episodes
# 7. Metric: track actions-to-win vs human baseline (RHAE proxy)

from arc_agi import Arcade, OperationMode
arc = Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode=None)  # headless for speed

# Train loop: collect rollouts on GPU, update policy, repeat
# Target: 50K env steps in <30 min on RTX 4090
```

---

## 2. Monte Carlo Tree Search with Learned Value Function

**Idea**: Use MCTS to plan action sequences. A small CNN on GPU evaluates board states (value function) to guide tree expansion. Local mode's 2000 FPS makes deep rollouts cheap.

**Why it might work**: MCTS excels at sequential decision problems with discrete actions. The GPU accelerates batch evaluation of leaf nodes during tree expansion.

**Fast-fail signal**: If MCTS doesn't solve level 1 of any game within 1000 rollouts, kill it.

**Implementation**:
```bash
uv add --script='exp02_mcts.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp02_mcts.py
# 1. Value network: 3-layer CNN (input: grid as tensor, output: scalar win probability)
# 2. MCTS: UCB1 selection, expand all 7 actions + grid of (x,y) for ACTION6
# 3. Batch leaf evaluation: collect 64 leaf states, forward pass on GPU in one batch
# 4. Rollout budget: 500 simulations per real action
# 5. After each game, train value net on (state, outcome) pairs from the tree
# 6. Run on ls20 first (likely simplest), then ft09, vc33

# Key optimization: env.step() is CPU-bound at ~2000 FPS
# GPU handles batched value estimation (64 states per forward pass ~0.5ms)
# Effective: ~1000 MCTS rollouts/second
```

---

## 3. Offline RL from Random Exploration Trajectories

**Idea**: Run the random agent for 100K episodes across all 3 games. Collect (state, action, reward, next_state) tuples. Train a Decision Transformer (GPU-intensive sequence model) on this offline dataset to learn from the random agent's accidental wins.

**Why it might work**: Even random play occasionally wins early levels. A transformer can learn "what did winning trajectories have in common?" without any online interaction.

**Fast-fail signal**: If the trained policy doesn't beat random agent's win rate after training on 100K episodes, kill it.

**Implementation**:
```bash
uv add --script='exp03_collect.py' 'arc-agi' 'numpy'
uv add --script='exp03_decision_transformer.py' 'torch' 'arc-agi' 'numpy' 'transformers'
```

```python
# Phase 1: exp03_collect.py — collect trajectories (CPU-bound, ~30 min)
# Run random agent OFFLINE, save trajectories as numpy arrays
# Schema: {states: (T,H,W), actions: (T,), rewards: (T,), returns_to_go: (T,)}

# Phase 2: exp03_decision_transformer.py — train on GPU
# Decision Transformer: context_length=50, n_layer=4, n_head=4, d_model=128
# Input: (return_to_go, state, action) triples
# Train: predict next action given (RTG, state_history)
# RTX 4090 handles batch_size=256, seq_len=50 easily (~10M params)
# Train for 100 epochs on collected data (~1 hour)
# Evaluate: does it win more than random?
```

---

## 4. Curiosity-Driven Exploration (ICM/RND)

**Idea**: ARC-AGI-3 measures *exploration* as a core capability. Use Intrinsic Curiosity Module (ICM) or Random Network Distillation (RND) to generate intrinsic rewards for visiting novel states. GPU trains the prediction networks in real-time.

**Why it might work**: Games require discovering rules through interaction. Curiosity bonuses push the agent to systematically try different actions in different states rather than repeating known patterns.

**Fast-fail signal**: If curiosity-driven agent doesn't discover more unique states than random agent after 10K episodes, kill it.

**Implementation**:
```bash
uv add --script='exp04_curiosity.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp04_curiosity.py
# RND approach (simpler than ICM):
# 1. Fixed random network f(state) -> 128-dim embedding (frozen)
# 2. Predictor network g(state) -> 128-dim embedding (trained)
# 3. Intrinsic reward = ||f(state) - g(state)||^2 (high for novel states)
# 4. Both networks: small CNNs on GPU
# 5. Policy: epsilon-greedy over actions, biased by curiosity
# 6. Track: unique_states_visited, levels_completed, actions_to_win
# 7. Compare directly to random baseline on same episode count

# GPU usage: forward pass both networks every step (~0.1ms)
# Can run at ~5000 decisions/sec with GPU inference
```

---

## 5. Program Synthesis via Genetic Programming on GPU

**Idea**: Represent agent strategies as small programs (if-then-else over grid features). Use GPU-accelerated parallel evaluation: run 1024 candidate programs simultaneously against the environment in batch.

**Why it might work**: ARC tasks often have simple underlying rules. GP can discover these rules as explicit programs, which generalize perfectly once found.

**Fast-fail signal**: If no program in the population solves level 1 within 100 generations, kill it.

**Implementation**:
```bash
uv add --script='exp05_gp.py' 'torch' 'arc-agi' 'numpy' 'deap'
```

```python
# exp05_gp.py
# DSL primitives: count_color(grid, c), find_pattern(grid, pattern),
#   click_at(x,y), neighbors(x,y), changed_cells(prev, curr)
# 1. Population of 1024 program trees
# 2. Each program: observe grid -> decide action sequence
# 3. GPU parallelism: encode all 1024 grid states as batch tensor
#    evaluate feature extraction primitives as batched torch ops
# 4. Fitness: -actions_to_win (lower is better), -inf if no win
# 5. Tournament selection, subtree crossover, point mutation
# 6. Run 100 generations, each gen evaluates 1024 programs x 3 episodes
# 7. RTX 4090 handles 1024 parallel feature extractions trivially

# The environment itself runs serially (CPU), so we evaluate programs
# one at a time but use GPU for the program's internal grid analysis
# Alternative: batch 8 envs across CPU cores with multiprocessing
```

---

## 6. Contrastive State Representation Learning

**Idea**: First learn a good state representation, then plan in that space. Use SimCLR-style contrastive learning: states from the same game/level that are temporally close should embed nearby; states from different games should be far apart. GPU trains the encoder.

**Why it might work**: A good representation makes downstream RL/planning much easier. 16-color grids have massive redundancy that a learned embedding can compress.

**Fast-fail signal**: If t-SNE of learned embeddings doesn't cluster by game/level after training, the representation is useless — kill it.

**Implementation**:
```bash
uv add --script='exp06_contrastive.py' 'torch' 'arc-agi' 'numpy' 'scikit-learn' 'matplotlib'
```

```python
# exp06_contrastive.py
# Phase 1: Collect 50K (state, game_id, level, timestep) tuples from random play
# Phase 2: Train CNN encoder with NT-Xent contrastive loss
#   Positive pairs: (state_t, state_t+1) from same episode
#   Negative pairs: states from different games
#   Encoder: ResNet-18 adapted for 16-channel input, output 64-dim
#   Batch size 512 on RTX 4090, train 50 epochs (~20 min)
# Phase 3: Evaluate representation quality
#   - t-SNE visualization (does it cluster?)
#   - Linear probe: can a linear classifier predict game_id from embedding?
#   - Use embedding as state for simple Q-learning, compare to raw grid input
```

---

## 7. World Model (DreamerV3-lite)

**Idea**: Learn a world model that predicts next grid state given current state + action. Then plan entirely in the model's imagination using the GPU. No environment interaction needed during planning.

**Why it might work**: ARC games are deterministic — a perfect world model enables perfect planning. Even an imperfect model can guide search. 2000 FPS provides fast training data.

**Fast-fail signal**: If world model prediction accuracy < 90% on held-out transitions after 1 hour of training, the model isn't learning game dynamics — kill it.

**Implementation**:
```bash
uv add --script='exp07_world_model.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp07_world_model.py
# World model: state_t + action_t -> state_t+1
# Architecture: U-Net style (encode grid, inject action, decode next grid)
#   Encoder: 4 conv layers (16->32->64->128)
#   Action embedding: learned 32-dim vector, broadcast and concatenated
#   Decoder: 4 deconv layers (128->64->32->16)
#   Output: per-cell softmax over 16 colors
#   ~5M params, fits easily on RTX 4090
#
# Training: 200K transitions from random play, batch_size=256
# Validation: hold out 20K transitions, measure per-cell accuracy
#
# Planning: given current state, BFS/DFS in imagined states
#   Expand top-K promising action sequences (beam search, width=64)
#   All 64 beams evaluated in single GPU batch
#   Search depth 20 actions, looking for predicted WIN state
```

---

## 8. Multi-Game Transfer via Meta-Learning (MAML)

**Idea**: ARC-AGI-3 tests generalization across games. Use Model-Agnostic Meta-Learning: inner loop adapts to a specific game in ~10 gradient steps, outer loop optimizes for fast adaptation across all games. GPU handles the second-order gradients.

**Why it might work**: The 5 evaluation dimensions (exploration, memory, etc.) suggest shared structure across games. Meta-learning explicitly optimizes for fast adaptation to new games.

**Fast-fail signal**: If meta-learned agent doesn't adapt to a held-out game faster than training from scratch after 5K outer steps, kill it.

**Implementation**:
```bash
uv add --script='exp08_maml.py' 'torch' 'arc-agi' 'numpy' 'higher'
```

```python
# exp08_maml.py
# Inner loop: 10 gradient steps of policy gradient on game G
# Outer loop: meta-gradient across all 3 games
# Policy: CNN encoder -> action head (same as exp01 but smaller, ~500K params)
#
# Per outer step:
#   For each game g in {ls20, ft09, vc33}:
#     Clone policy, run 10 episodes, compute PG loss, 10 inner updates
#     Evaluate adapted policy for 5 episodes (meta-validation)
#   Meta-loss = sum of meta-validation losses
#   Update meta-parameters
#
# RTX 4090 needed for: second-order gradients via `higher` library
# ~2x memory of standard training, still fits in 24GB easily
# Target: 5K outer steps in ~2 hours
```

---

## 9. GPU-Accelerated Beam Search with Action Embedding Similarity

**Idea**: Embed all possible actions into a learned space. At each step, encode the current grid state, and find the K nearest action embeddings. Execute top-K actions in parallel (K environment copies), keep the best trajectories. Pure search, no RL.

**Why it might work**: Brute-force works when you can evaluate fast enough. RTX 4090 can score thousands of candidate actions per step via batch dot products. Local mode at 2000 FPS means you can actually execute the top candidates.

**Fast-fail signal**: If beam search with K=64 doesn't solve level 1 of ls20 within 200 steps, kill it.

**Implementation**:
```bash
uv add --script='exp09_beam_search.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp09_beam_search.py
# Action space: 6 simple actions + ACTION6 with (x,y) discretized to grid
# For 64x64 grid: 6 + 4096 = 4102 possible actions per step
#
# 1. State encoder: CNN -> 256-dim (on GPU)
# 2. Action embeddings: 4102 learned vectors of dim 256 (on GPU)
# 3. Score all actions: state_embed @ action_embeds.T -> (4102,) scores
#    Single matmul on GPU: <0.01ms
# 4. Top-K=64 actions selected
# 5. Execute top-64 in 64 parallel env copies (multiprocessing, CPU)
#    At 2000 FPS each: 64 steps in ~32ms
# 6. Keep best 16 trajectories (by heuristic: most grid changes = progress)
# 7. Repeat up to depth 200
#
# Bootstrap action embeddings: random init, then update based on
# which actions led to grid changes (gradient-free, just moving averages)
```

---

## 10. Hybrid LLM + GPU Vision: Claude Reasons, CNN Perceives

**Idea**: Use a local GPU-accelerated vision model (small CNN or CLIP-like encoder) to describe the grid state in structured text, then feed that description to Claude (via API) for high-level reasoning about what to do. The GPU handles perception at 1000+ FPS; the LLM handles strategy.

**Why it might work**: LLMs are strong at reasoning about rules from descriptions but weak at raw grid perception. CNNs are strong at pattern detection but weak at strategic reasoning. Combine both.

**Fast-fail signal**: If the hybrid doesn't beat the built-in `--agent=llm` baseline on any game after 50 episodes, kill it.

**Implementation**:
```bash
uv add --script='exp10_hybrid.py' 'torch' 'arc-agi' 'anthropic' 'numpy'
```

```python
# exp10_hybrid.py
# GPU perception module (runs locally, no API cost):
#   - CNN trained on random play data to detect:
#     * color clusters/regions
#     * symmetry axes
#     * repeating patterns
#     * changed cells since last action
#   - Outputs structured JSON: {"regions": [...], "symmetry": "horizontal",
#     "changed_cells": 5, "dominant_colors": [2,7], "pattern": "grid_2x2"}
#
# LLM reasoning module (Claude API):
#   - System prompt: "You are solving an ARC-AGI puzzle. Given structured
#     observations, decide the next action."
#   - Receives: perception JSON + action history + reward history
#   - Returns: action choice with brief reasoning
#   - Call only every N steps (e.g., every 5) to manage API costs
#   - Between LLM calls, use a local heuristic (repeat last action type,
#     or follow LLM's multi-step plan)
#
# GPU cost: ~1ms per frame for CNN perception
# API cost: ~$0.01 per LLM call, ~50 calls per game = $0.50/game
# Compare to: --agent=llm (calls LLM every step, no vision preprocessing)
```

---

## Quick-Start: Running All Experiments

```bash
# Install base deps
uv add --script='run_all.py' 'arc-agi'

# Run experiments sequentially with timeout (kill after 2 hours each)
for i in $(seq -w 1 10); do
  echo "=== Experiment $i ==="
  timeout 7200 uv run exp${i}_*.py --game=ls20 2>&1 | tee logs/exp${i}.log
done
```

## Evaluation Harness

Every experiment should output a standard JSON line per episode:
```json
{"exp": 1, "game": "ls20", "level": 1, "actions": 42, "won": true, "time_s": 1.3}
```

Compare to random baseline:
```bash
uv run main.py --agent=random --game=ls20  # get baseline actions-to-win
```

RHAE proxy: `min(1.0, (human_baseline / your_actions) ** 2)`

## Priority Order (highest expected signal first)

1. **Exp 7** (World Model) — deterministic games reward accurate models
2. **Exp 2** (MCTS) — proven on similar problems, GPU accelerates leaf eval
3. **Exp 4** (Curiosity) — directly targets ARC's exploration dimension
4. **Exp 10** (Hybrid LLM+CNN) — leverages strongest reasoning available
5. **Exp 1** (ViT+PPO) — standard RL baseline, good calibration point
6. **Exp 3** (Offline RL) — cheap to collect data, transformer might find signal
7. **Exp 9** (Beam Search) — pure search, no training, immediate signal
8. **Exp 6** (Contrastive) — representation quality is measurable fast
9. **Exp 8** (MAML) — highest potential but most complex to debug
10. **Exp 5** (GP) — wild card, could find exact solutions or nothing
