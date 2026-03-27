# The Three Best ARC-AGI-3 Approaches

**Hardware**: RTX 4090 (24GB VRAM), PyTorch 2.8.0+cu128, CUDA 12.4
**Core insight from the critique**: ARC-AGI-3 requires **interactive hypothesis formation** — not policy learning, not search speed, not representation tricks. The agent must act like an experimental scientist: observe, hypothesize, test, revise, and only then exploit.

These three ideas synthesize the strongest elements from the original ten and the counter-proposals, filtered through what actually matters: exploration, memory, goal acquisition, and generalization to unseen games.

---

## Idea 1: LLM Scientist Agent with Local Model + Verified Hypotheses

**Estimated probability of solving early levels: 40-50%**

### What it is

An LLM runs a deliberate scientific loop: observe the grid, propose hypotheses about what actions do and what the goal might be, design experiments (specific actions) to discriminate between hypotheses, execute them, update beliefs, and only switch to exploitation once confidence is high. The RTX 4090 runs a local open-weight LLM (Qwen2.5-32B-AWQ or Llama-3.1-8B-Instruct) so you can iterate thousands of times without API cost. Claude API is reserved for hard cases or as a "second opinion" when the local model gets stuck.

### Why this is the best idea

- **Directly targets all 5 evaluation dimensions**: exploration (deliberate probes), percept-plan-action (the full loop), memory (hypothesis log), goal acquisition (Bayesian goal narrowing), alignment (self-correcting via verification).
- **LLMs already encode vast world knowledge** about grids, puzzles, spatial reasoning, games, and cause-and-effect — no training needed.
- **Verification prevents hallucination**: the LLM proposes, but the environment confirms. A wrong hypothesis gets disproven and pruned, not blindly followed.
- **Generalizes by design**: the same scientific loop works on any game because it discovers rules from scratch each time.
- **GPU makes it cheap**: Qwen2.5-32B-AWQ at 4-bit runs at ~40 tok/s on RTX 4090, fast enough for 50+ hypothesis cycles per game episode. No API cost means you can run 1000+ episodes to tune the prompt and loop structure.

### Why it might fail

- Local 8B-32B models may lack the reasoning depth to propose non-obvious hypotheses.
- Prompt engineering is critical — a bad system prompt makes the agent ramble instead of reason.
- The structured state serialization must be tight; too much noise and the LLM loses focus.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                 SCIENTIST LOOP                   │
│                                                  │
│  1. OBSERVE: serialize grid as structured text   │
│     - Object regions (connected components)      │
│     - Color histogram                            │
│     - Delta from previous state                  │
│     - Available actions for this game            │
│                                                  │
│  2. HYPOTHESIZE: LLM proposes/updates beliefs    │
│     - "ACTION1 probably moves the red object"    │
│     - "Goal might be: clear all blue cells"      │
│     - Confidence scores per hypothesis           │
│                                                  │
│  3. DESIGN EXPERIMENT: LLM picks action that     │
│     maximally discriminates between hypotheses   │
│     - "If ACTION1 moves red, clicking (3,4)      │
│       should shift it. If not, hypothesis dead." │
│                                                  │
│  4. EXECUTE: run action in environment           │
│                                                  │
│  5. VERIFY: compare prediction to actual result  │
│     - Prune disproven hypotheses                 │
│     - Strengthen confirmed ones                  │
│     - Log (state, action, effect) to memory      │
│                                                  │
│  6. DECIDE: if high confidence in goal + rules,  │
│     switch to EXPLOIT mode (plan optimal path).  │
│     Otherwise, loop back to step 2.              │
└─────────────────────────────────────────────────┘
```

### GPU usage

The RTX 4090 serves the local LLM. This is the single most valuable use of the GPU for ARC-AGI-3 — it turns every LLM call from $0.01 and 2 seconds into $0.00 and 0.5 seconds, enabling:

- **Prompt iteration**: test 50 system prompt variants across 3 games in an afternoon
- **Ablation studies**: does verification matter? does structured perception help? remove each component and measure
- **Scale**: run 1000 episodes per game to get statistically significant win rates

### Implementation

```bash
# Install vllm for fast local LLM serving on RTX 4090
uv add --script='scientist_agent.py' 'arc-agi' 'openai' 'numpy'
```

```bash
# Start local LLM server (separate terminal, runs on GPU)
# Qwen2.5-32B-AWQ fits in 24GB VRAM at 4-bit
uv run --with vllm vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.90
```

```python
# scientist_agent.py
# /// script
# dependencies = ["arc-agi", "openai", "numpy"]
# ///

"""
Scientist Agent: hypothesis-driven interactive reasoning for ARC-AGI-3.
Uses local LLM via vLLM (OpenAI-compatible API on localhost:8000).
Falls back to Claude API for hard cases if ANTHROPIC_API_KEY is set.
"""

import json
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from openai import OpenAI

# --- Config ---
LOCAL_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LOCAL_URL = "http://localhost:8000/v1"
MAX_PROBE_STEPS = 30      # max steps in exploration phase
MAX_EXPLOIT_STEPS = 50     # max steps in exploitation phase
CONFIDENCE_THRESHOLD = 0.8 # switch to exploit when belief confidence > this

# --- Grid Perception (CPU, no ML needed) ---
def serialize_grid(grid: np.ndarray, prev_grid: np.ndarray = None) -> dict:
    """Convert raw grid to structured observation dict."""
    h, w = grid.shape
    obs = {
        "grid_size": f"{w}x{h}",
        "color_counts": {},
        "objects": [],      # connected components with bounding boxes
        "delta": None,      # cells that changed since last step
    }
    # Color histogram
    unique, counts = np.unique(grid, return_counts=True)
    obs["color_counts"] = {int(c): int(n) for c, n in zip(unique, counts)}

    # Connected components (flood fill) for each non-background color
    background = int(np.argmax(counts))  # most common color = background
    visited = np.zeros_like(grid, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not visited[y, x] and grid[y, x] != background:
                color = int(grid[y, x])
                # BFS flood fill
                component = []
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if 0 <= cx < w and 0 <= cy < h and not visited[cy, cx] and grid[cy, cx] == color:
                        visited[cy, cx] = True
                        component.append((cx, cy))
                        stack.extend([(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)])
                if len(component) >= 2:  # skip single pixels
                    xs, ys = zip(*component)
                    obs["objects"].append({
                        "color": color, "size": len(component),
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "center": [round(np.mean(xs),1), round(np.mean(ys),1)],
                    })

    # Delta from previous state
    if prev_grid is not None:
        changed = np.argwhere(grid != prev_grid)
        obs["delta"] = {
            "num_changed": len(changed),
            "changed_cells": [(int(c[1]), int(c[0])) for c in changed[:20]],  # cap at 20
        }
    return obs

# --- Hypothesis Memory ---
class BeliefState:
    def __init__(self):
        self.action_hypotheses = {}  # action -> list of {rule, confidence, evidence}
        self.goal_hypotheses = []    # list of {description, confidence, evidence}
        self.action_log = []         # list of {step, action, obs_before, obs_after, effect}

    def to_prompt_text(self) -> str:
        lines = ["## Current Beliefs"]
        lines.append("### Action effects:")
        for action, hyps in self.action_hypotheses.items():
            for h in hyps:
                lines.append(f"  - {action}: \"{h['rule']}\" (confidence: {h['confidence']:.0%}, evidence: {h['evidence']})")
        lines.append("### Goal hypotheses:")
        for h in self.goal_hypotheses:
            lines.append(f"  - \"{h['description']}\" (confidence: {h['confidence']:.0%}, evidence: {h['evidence']})")
        lines.append(f"### Actions taken so far: {len(self.action_log)}")
        if self.action_log:
            for entry in self.action_log[-5:]:  # show last 5
                lines.append(f"  Step {entry['step']}: {entry['action']} -> {entry['effect']}")
        return "\n".join(lines)

# --- LLM Interface ---
client = OpenAI(base_url=LOCAL_URL, api_key="not-needed")

def ask_llm(system: str, user: str, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=1024,
    )
    return resp.choices[0].message.content

SYSTEM_PROMPT = """You are a scientist agent solving an unknown interactive puzzle on a 2D grid.

Your job is to discover the rules of the environment and the win condition through
deliberate experimentation, then solve the puzzle efficiently.

PHASE 1 - PROBE: Design experiments to learn what each action does and what the goal is.
Pick actions that maximally discriminate between your hypotheses.

PHASE 2 - EXPLOIT: Once you're confident in the rules and goal, execute the
shortest path to win.

Always respond in this exact JSON format:
{
  "phase": "probe" or "exploit",
  "reasoning": "one sentence about what you expect to learn or achieve",
  "action": "RESET" | "ACTION1" | "ACTION2" | "ACTION3" | "ACTION4" | "ACTION5" | "ACTION6",
  "x": null or integer (required only for ACTION6),
  "y": null or integer (required only for ACTION6),
  "updated_hypotheses": {
    "action_effects": [{"action": "...", "rule": "...", "confidence": 0.0-1.0}],
    "goals": [{"description": "...", "confidence": 0.0-1.0}]
  }
}"""

# --- Main Loop ---
def run_episode(game_id: str):
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(game_id, render_mode="terminal")
    obs = env.reset()

    belief = BeliefState()
    prev_grid = None
    total_actions = 0

    for step in range(MAX_PROBE_STEPS + MAX_EXPLOIT_STEPS):
        grid = np.array(obs.grid)  # adapt to actual API
        structured_obs = serialize_grid(grid, prev_grid)

        user_msg = f"## Observation (step {step})\n```json\n{json.dumps(structured_obs, indent=2)}\n```\n\n{belief.to_prompt_text()}\n\nWhat action should I take and why?"

        response_text = ask_llm(SYSTEM_PROMPT, user_msg)
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: try to extract action from text
            decision = {"action": "ACTION1", "reasoning": "parse error fallback"}

        # Execute action
        action_name = decision.get("action", "ACTION1")
        action = getattr(GameAction, action_name, GameAction.ACTION1)
        data = {}
        if action_name == "ACTION6" and decision.get("x") is not None:
            data = {"x": decision["x"], "y": decision["y"]}

        obs = env.step(action, data=data) if data else env.step(action)
        total_actions += 1

        # Record effect
        new_grid = np.array(obs.grid)
        effect_obs = serialize_grid(new_grid, grid)
        belief.action_log.append({
            "step": step, "action": action_name,
            "effect": f"{effect_obs.get('delta', {}).get('num_changed', 0)} cells changed",
            "obs_before": structured_obs, "obs_after": effect_obs,
        })

        # Update hypotheses from LLM response
        if "updated_hypotheses" in decision:
            uh = decision["updated_hypotheses"]
            for ae in uh.get("action_effects", []):
                belief.action_hypotheses.setdefault(ae["action"], [])
                # Update or add
                found = False
                for existing in belief.action_hypotheses[ae["action"]]:
                    if existing["rule"] == ae["rule"]:
                        existing["confidence"] = ae["confidence"]
                        existing["evidence"] = f"step {step}"
                        found = True
                if not found:
                    belief.action_hypotheses[ae["action"]].append({
                        "rule": ae["rule"], "confidence": ae["confidence"], "evidence": f"step {step}"
                    })
            belief.goal_hypotheses = [
                {"description": g["description"], "confidence": g["confidence"], "evidence": f"step {step}"}
                for g in uh.get("goals", [])
            ]

        prev_grid = grid

        # Check win
        if obs.state == "WIN":  # adapt to actual API
            print(json.dumps({"game": game_id, "won": True, "actions": total_actions}))
            return True, total_actions

        if obs.state == "GAME_OVER":
            print(json.dumps({"game": game_id, "won": False, "actions": total_actions}))
            return False, total_actions

    return False, total_actions

if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    won, actions = run_episode(game)
    print(f"Result: {'WIN' if won else 'LOSS'} in {actions} actions")
```

### Fast-fail criteria

| Check | Threshold | When |
|-------|-----------|------|
| LLM produces parseable JSON | >80% of steps | After 10 episodes |
| Hypotheses converge (confidence >0.7) | Within 15 probe steps | After 10 episodes |
| Beats random agent win rate | Any margin | After 50 episodes |
| Solves level 1 of any free game | At least 1 game | After 100 episodes |

### Tuning levers (things to vary quickly)

- Swap local model: 8B (fast, dumb) vs 32B (slow, smarter) vs Claude API (expensive, strongest)
- Adjust probe budget: 10 vs 30 vs 50 steps before forcing exploit
- Grid serialization detail: minimal (histogram + delta only) vs full (objects + spatial relations)
- Temperature: 0.1 (deterministic) vs 0.5 (diverse hypotheses)

---

## Idea 2: Object-Centric World Model + Information-Gain Search

**Estimated probability of solving early levels: 30-40%**

### What it is

A two-stage system. **Stage 1**: a GPU-trained CNN segments each grid frame into objects (connected regions of the same color) and learns to predict how objects change when actions are applied — not pixel-by-pixel, but object-by-object (moved? deleted? color changed? merged?). **Stage 2**: a search algorithm picks actions that maximize **information gain** about the game's hidden rules, not just actions that change the grid. Once the model is confident, it switches to goal-directed planning in the learned object-transition space.

### Why this is the second-best idea

- **Object-centric models generalize better than pixel models**: if you learn "ACTION1 moves objects right," that transfers across grid sizes and layouts. A U-Net pixel predictor memorizes specific grids.
- **Information-gain search is the principled version of curiosity**: instead of "visit novel states" (RND), it asks "which action teaches me the most about how this game works?" This directly addresses ARC's exploration dimension.
- **The GPU is used for its actual strength**: batch training and batch inference of the transition model, not just as an expensive random number generator. Evaluating 64 candidate actions through the model in one forward pass takes <1ms.
- **Deterministic games mean the model can be exact**: unlike Atari or robotics where dynamics are stochastic, ARC-AGI-3 games are deterministic. A well-trained model can predict transitions perfectly, enabling perfect planning.

### Why it might fail

- Object segmentation errors (splitting one object into two, or merging two) poison downstream predictions.
- Some games may not have clean object structure — the grid might encode state in patterns, textures, or implicit counters.
- Needs ~50K transitions of random play data to train, and if random play doesn't cover enough of the state space, the model has blind spots.

### Architecture

```
┌──────────────────────────────────────────────────┐
│           OBJECT-CENTRIC WORLD MODEL             │
│                                                   │
│  Grid Frame ──> Object Segmenter (GPU)            │
│     64x64x16       │                              │
│                     ▼                              │
│     Object List: [{id, color, bbox, pixels}, ...] │
│                     │                              │
│     + Action ───────┤                              │
│                     ▼                              │
│     Object Transition Predictor (GPU)              │
│     Per object: {moved_to, color_changed_to,       │
│                  deleted, merged_with, spawned}     │
│                     │                              │
│                     ▼                              │
│     Predicted Next Object List                     │
│     ──> Render back to predicted grid              │
│     ──> Compare to actual grid (training signal)   │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│         INFORMATION-GAIN SEARCH                   │
│                                                   │
│  For each candidate action a:                     │
│    1. Predict next state via world model           │
│    2. Compute information gain:                    │
│       IG(a) = H(beliefs_before) - E[H(beliefs     │
│               _after | predicted_outcome(a))]      │
│    3. Blend with goal score:                       │
│       score(a) = α·IG(a) + (1-α)·goal_progress(a)│
│                                                   │
│  α starts at 1.0 (pure exploration) and decays    │
│  toward 0.0 (pure exploitation) as beliefs        │
│  converge.                                        │
│                                                   │
│  All candidate actions scored in one GPU batch.    │
│  Top action executed. Model updated online.        │
└──────────────────────────────────────────────────┘
```

### GPU usage

| Component | GPU memory | Throughput |
|-----------|-----------|------------|
| Object segmenter (3-layer CNN) | ~50MB | 10K grids/sec |
| Transition predictor (per-object MLP + attention) | ~200MB | 5K predictions/sec |
| Batch action scoring (64 candidates) | ~10MB | <1ms per step |
| Training (online, after each real step) | ~500MB | 1 gradient step in <5ms |
| **Total** | **<1GB** | Leaves 23GB free for model scaling |

### Implementation

```bash
uv add --script='exp_worldmodel_collect.py' 'arc-agi' 'numpy'
uv add --script='exp_worldmodel_train.py' 'torch' 'numpy'
uv add --script='exp_worldmodel_agent.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp_worldmodel_collect.py — Phase 1: gather transitions (~20 min)
# Run random agent OFFLINE on all 3 games, 50K steps each
# Save: (grid_t, action_t, grid_t+1) as compressed numpy
# Also save: object-segmented versions (connected components)

# exp_worldmodel_train.py — Phase 2: train models (~30 min on GPU)
# Object segmenter: input 64x64x16 one-hot, output per-cell object ID
#   Trained with self-supervision: connected components as labels
#   3-layer CNN, ~500K params
# Transition predictor: input (object_features, action_onehot)
#   output (delta_x, delta_y, new_color, deleted_prob)
#   Per-object MLP with cross-object attention (which objects interact?)
#   ~2M params, trained on collected transitions
# Validation: hold out 10K transitions, measure prediction accuracy
#   KILL if object-level prediction accuracy < 85%

# exp_worldmodel_agent.py — Phase 3: deploy agent
# At each step:
#   1. Segment current grid into objects (GPU, <0.1ms)
#   2. For each of ~100 candidate actions (6 simple + top-94 (x,y) positions):
#      Predict next state via transition model (GPU batch, <1ms total)
#   3. Score each action by information gain:
#      - Maintain belief distribution over "action effect types"
#        (e.g., P(ACTION1=move_right)=0.6, P(ACTION1=delete)=0.2, ...)
#      - IG = entropy reduction in this distribution
#   4. Pick argmax(α*IG + (1-α)*goal_progress)
#   5. Execute, observe, update model online (1 gradient step)
```

### Fast-fail criteria

| Check | Threshold | When |
|-------|-----------|------|
| Object segmentation matches connected components | >95% IOU | After training |
| Transition prediction accuracy (object-level) | >85% | After training |
| Information-gain actions explore faster than random | Measurable | After 20 episodes |
| Agent solves level 1 of any game | At least 1 | After 100 episodes |

---

## Idea 3: Two-Phase Probe-Solve Agent with Causal Memory

**Estimated probability of solving early levels: 30-38%**

### What it is

A hard architectural split between **probing** (learning the rules) and **solving** (winning efficiently). Phase 1 runs a battery of systematic diagnostic actions — poke each object, try each action type, check for symmetries, test boundaries — and records every (state, action, effect) triple in a structured causal memory. Phase 2 retrieves relevant memories, constructs a plan, and executes it. The GPU accelerates memory encoding, similarity search, and pattern detection.

### Why this is the third-best idea

- **Matches how humans actually solve ARC puzzles**: people poke around first ("what does this button do?"), build a mental model, then solve. This is exactly probe-then-solve.
- **Causal memory transfers across games**: if you learned "clicking a colored cell removes it" in game A, and game B has similar colored cells, retrieval finds the relevant memory and you skip re-probing. This is the only approach of the three that explicitly builds cross-game transfer.
- **Probe quality is separately measurable**: you can evaluate "did probing discover all action effects?" independently of "did solving use them well?" This makes debugging much faster than end-to-end approaches.
- **GPU-accelerated memory retrieval**: embed (state, action, effect) triples using a small CNN encoder on GPU, retrieve similar experiences via cosine similarity in <1ms. With 100K+ memories from many games, this becomes a genuine knowledge base.

### Why it might fail

- The probe battery might waste too many actions on uninformative tests, leaving too few for solving (RHAE penalizes high action counts).
- Deciding when to stop probing is hard — too early misses critical rules, too late wastes the action budget.
- Cross-game memory retrieval might return false matches (similar-looking grids with different mechanics).

### Architecture

```
┌────────────────────────────────────────────────────┐
│              PHASE 1: PROBE                         │
│                                                     │
│  Systematic Diagnostic Battery:                     │
│  1. Try each ACTION1-5 once from initial state      │
│     Record: what changed? what didn't?              │
│  2. For ACTION6: click on each distinct object      │
│     (one click per color region, ~5-10 probes)      │
│  3. Try ACTION7 (undo) after each action            │
│     → Learn: is undo available? what does it undo?  │
│  4. Repeat an action twice                          │
│     → Learn: is the effect consistent? cumulative?  │
│  5. Try action on empty space vs on object           │
│     → Learn: does target matter?                    │
│                                                     │
│  Total probe budget: 15-25 actions                  │
│  Each probe stored as structured triple:             │
│  (state_pattern, action, effect_pattern, game_id)   │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│         CAUSAL MEMORY (GPU-accelerated)             │
│                                                     │
│  Storage: list of {                                 │
│    state_embedding: 128-dim (CNN encoder on GPU),   │
│    action: enum,                                    │
│    effect: {cells_changed, objects_moved,            │
│            objects_deleted, score_delta},            │
│    game_id: str,                                    │
│    confidence: float (how many times confirmed)     │
│  }                                                  │
│                                                     │
│  Retrieval: given new state, find K nearest         │
│  state_embeddings via batch cosine sim on GPU       │
│  Return: relevant action-effect memories            │
│                                                     │
│  Cross-game: memories from game A retrieved when    │
│  game B has similar state patterns                  │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│         PHASE 2: SOLVE                              │
│                                                     │
│  1. Retrieve relevant memories for current state    │
│  2. Infer goal from probe observations:             │
│     - Score increased when X happened → goal is X   │
│     - State "WIN" triggered after Y → Y is target   │
│  3. Construct action plan using inferred rules:     │
│     BFS/DFS over abstract action sequences          │
│     (not raw grid states — over effect chains)      │
│  4. Execute plan, re-plan if unexpected result      │
│                                                     │
│  Solve budget: remaining actions after probing      │
│  Target: match or beat human action count           │
└────────────────────────────────────────────────────┘
```

### GPU usage

| Component | GPU memory | Purpose |
|-----------|-----------|---------|
| State encoder (CNN -> 128-dim) | ~30MB | Embed grids for memory storage and retrieval |
| Memory index (100K embeddings) | ~50MB | Cosine similarity search over all stored experiences |
| Pattern detector (connected components, symmetry, repetition) | ~100MB | Structured perception for probe analysis |
| Goal inference network (optional, small classifier) | ~20MB | Predict goal type from probe observations |
| **Total** | **~200MB** | Almost all VRAM free for model scaling |

### Implementation

```bash
uv add --script='exp_probe_solve.py' 'torch' 'arc-agi' 'numpy'
```

```python
# exp_probe_solve.py
# /// script
# dependencies = ["torch", "arc-agi", "numpy"]
# ///

"""
Two-Phase Probe-Solve Agent with Causal Memory for ARC-AGI-3.
GPU used for: state encoding, memory retrieval, pattern detection.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

# --- State Encoder (GPU) ---
class StateEncoder(nn.Module):
    """Encode 64x64 grid with 16 colors into 128-dim embedding."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)

# --- Causal Memory ---
class CausalMemory:
    def __init__(self, encoder, device="cuda"):
        self.encoder = encoder
        self.device = device
        self.entries = []          # list of dicts
        self.embeddings = None     # (N, 128) tensor on GPU

    def store(self, grid_before, action, effect_summary, game_id):
        emb = self._encode(grid_before)
        self.entries.append({
            "action": action, "effect": effect_summary,
            "game_id": game_id, "confirmed_count": 1,
        })
        if self.embeddings is None:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)])

    def retrieve(self, grid, k=10):
        if self.embeddings is None or len(self.entries) == 0:
            return []
        query = self._encode(grid).unsqueeze(0)  # (1, 128)
        sims = torch.mm(query, self.embeddings.T).squeeze(0)  # (N,)
        topk = torch.topk(sims, min(k, len(self.entries)))
        return [(self.entries[i], sims[i].item()) for i in topk.indices.tolist()]

    def _encode(self, grid):
        # One-hot encode grid to (16, H, W), pad to 64x64
        h, w = grid.shape
        onehot = np.zeros((16, 64, 64), dtype=np.float32)
        for c in range(16):
            onehot[c, :h, :w] = (grid == c).astype(np.float32)
        t = torch.from_numpy(onehot).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(t).squeeze(0)

# --- Probe Battery ---
SIMPLE_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5]

def run_probe_battery(env, memory, game_id):
    """Systematic diagnostic probes. Returns action-effect log."""
    log = []
    obs = env.reset()

    def do_action(action, data=None):
        """Execute action, record effect, return new obs."""
        grid_before = np.array(obs.grid)
        new_obs = env.step(action, data=data) if data else env.step(action)
        grid_after = np.array(new_obs.grid)
        delta = int(np.sum(grid_before != grid_after))
        effect = {
            "cells_changed": delta,
            "score_delta": getattr(new_obs, 'score', 0) - getattr(obs, 'score', 0),
            "state": str(new_obs.state),
        }
        memory.store(grid_before, str(action), effect, game_id)
        log.append({"action": str(action), "data": data, "effect": effect})
        return new_obs, effect

    # Probe 1: Try each simple action from initial state
    initial_obs = obs
    for action in SIMPLE_ACTIONS:
        obs = env.reset()  # fresh start for each test
        obs, effect = do_action(action)

    # Probe 2: Click on distinct objects (ACTION6)
    obs = env.reset()
    grid = np.array(obs.grid)
    # Find centroids of each color region
    unique_colors = np.unique(grid)
    bg = int(np.argmax(np.bincount(grid.flatten())))
    for color in unique_colors:
        if color == bg:
            continue
        ys, xs = np.where(grid == int(color))
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            obs = env.reset()
            obs, effect = do_action(GameAction.ACTION6, data={"x": cx, "y": cy})

    # Probe 3: Test undo (ACTION7)
    obs = env.reset()
    obs, _ = do_action(GameAction.ACTION1)
    obs, _ = do_action(GameAction.ACTION7)

    # Probe 4: Repeat same action twice (test consistency)
    obs = env.reset()
    obs, effect1 = do_action(GameAction.ACTION1)
    obs, effect2 = do_action(GameAction.ACTION1)

    return log

# --- Solver ---
def solve_phase(env, memory, probe_log, game_id, max_steps=50):
    """Use inferred rules to solve the puzzle efficiently."""
    obs = env.reset()
    actions_taken = 0

    for step in range(max_steps):
        grid = np.array(obs.grid)

        # Retrieve relevant memories
        relevant = memory.retrieve(grid, k=5)

        # Simple heuristic solver: pick action with highest score_delta
        best_action = None
        best_score = -float('inf')
        for entry, sim in relevant:
            sd = entry["effect"].get("score_delta", 0)
            if sd > best_score:
                best_score = sd
                best_action = entry["action"]

        # Fallback: try each action, pick one that changes the grid most
        if best_action is None or best_score <= 0:
            best_action = "GameAction.ACTION1"  # default

        # Execute (simplified — parse action name back to enum)
        action = GameAction.ACTION1  # would parse best_action properly
        obs = env.step(action)
        actions_taken += 1

        if obs.state == "WIN":
            return True, actions_taken
        if obs.state == "GAME_OVER":
            return False, actions_taken

    return False, actions_taken

# --- Main ---
if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = StateEncoder().to(device)
    memory = CausalMemory(encoder, device)

    games = sys.argv[1:] if len(sys.argv) > 1 else ["ls20", "ft09", "vc33"]

    for game_id in games:
        arc = Arcade(operation_mode=OperationMode.OFFLINE)
        env = arc.make(game_id, render_mode=None)

        # Phase 1: Probe
        probe_log = run_probe_battery(env, memory, game_id)
        print(f"[{game_id}] Probe complete: {len(probe_log)} actions, {len(memory.entries)} memories")

        # Phase 2: Solve
        won, actions = solve_phase(env, memory, probe_log, game_id)
        print(json.dumps({"game": game_id, "won": won, "actions": actions,
                          "probe_actions": len(probe_log), "memories": len(memory.entries)}))
```

### Fast-fail criteria

| Check | Threshold | When |
|-------|-----------|------|
| Probes discover distinct effects for >3 actions | Per game | After first run |
| Memory retrieval returns relevant hits cross-game | >50% precision | After 3 games |
| Solver beats random on any game | Any margin | After 50 episodes |
| Probe + solve total actions < 2x random's winning actions | On winning episodes | After 100 episodes |

---

## Comparison and Recommendation

| | Idea 1: LLM Scientist | Idea 2: Object World Model | Idea 3: Probe-Solve Memory |
|-|----------------------|---------------------------|---------------------------|
| **P(early levels)** | 40-50% | 30-40% | 30-38% |
| **GPU VRAM used** | 20-24GB (local LLM) | <1GB (leaves room) | <500MB (very light) |
| **Time to first signal** | ~1 hour (prompt + 10 episodes) | ~2 hours (collect + train + test) | ~30 min (direct probe-solve) |
| **Cross-game transfer** | Via LLM world knowledge | Via shared object features | Via causal memory retrieval |
| **Biggest risk** | Local LLM too dumb | Object segmentation errors | Probe budget waste |
| **Best combined with** | Idea 3's probe structure | Idea 1's LLM for goal inference | Idea 2's object perception |

### The real play: combine all three

The strongest possible agent would use:
- **Idea 3's probe-solve structure** as the outer loop (systematic, measurable)
- **Idea 2's object-centric perception** as the grid understanding layer
- **Idea 1's LLM** as the hypothesis generator and goal inferencer

But start with **Idea 1** alone — it has the highest standalone probability, the fastest iteration cycle, and the most tuning levers. If the local LLM can reason about grid states at all, you'll see signal within an hour.

---

# Implementation Spec: Continuous Research Harness

Inspired by the [autoresearch](~/p/autoresearch) project's pattern of autonomous, continuous experimentation with keep/discard tracking, fixed-time budgets, and never-stop looping — adapted for ARC-AGI-3's multi-experiment, multi-game structure.

## Directory Layout

```
arc-agi-3/
├── CLAUDE.md                  # Project instructions (already exists)
├── three.md                   # This spec
├── ten.md                     # Original ideas (reference)
├── COUNTER.md                 # Critique (reference)
├── harness.py                 # Main orchestrator — round-robin runner
├── db.py                      # SQLite state manager
├── notify.py                  # iMessage milestone alerts
├── arc_agi_3.db               # SQLite database (auto-created)
├── shared/                    # Shared utilities across experiments
│   ├── __init__.py
│   ├── perception.py          # Grid serialization, connected components, delta
│   ├── metrics.py             # RHAE calculation, result formatting
│   └── env_wrapper.py         # Gymnasium-style wrapper for arc_agi environments
├── exp1_scientist/            # Idea 1: LLM Scientist Agent
│   ├── agent.py               # Main agent logic
│   ├── prompts.py             # System prompts, structured output templates
│   ├── belief.py              # BeliefState / hypothesis tracking
│   ├── config.py              # Tuning knobs (model, temperature, probe budget)
│   ├── results/               # Auto-created, per-run result JSONLs
│   │   └── 2026-03-27_001.jsonl
│   └── checkpoints/           # Saved belief states, best prompts
│       └── best_prompt.json
├── exp2_worldmodel/           # Idea 2: Object-Centric World Model
│   ├── agent.py               # Agent using trained model for planning
│   ├── collect.py             # Transition data collection (random play)
│   ├── train.py               # Train segmenter + transition predictor
│   ├── models.py              # CNN segmenter, object transition MLP
│   ├── config.py              # Hyperparameters
│   ├── results/
│   │   └── 2026-03-27_001.jsonl
│   ├── checkpoints/           # Model weights (.pt files)
│   │   ├── segmenter_v1.pt
│   │   └── transition_v1.pt
│   └── data/                  # Collected transitions (numpy)
│       └── ls20_transitions.npz
└── exp3_probe_solve/          # Idea 3: Probe-Solve with Causal Memory
    ├── agent.py               # Main agent logic
    ├── probe.py               # Systematic diagnostic battery
    ├── memory.py              # CausalMemory with GPU embeddings
    ├── solver.py              # Plan construction + execution
    ├── encoder.py             # StateEncoder CNN
    ├── config.py              # Tuning knobs
    ├── results/
    │   └── 2026-03-27_001.jsonl
    └── checkpoints/           # Encoder weights, serialized memory
        ├── encoder_v1.pt
        └── memory_v1.pkl
```

## SQLite State Database (`arc_agi_3.db`)

Central source of truth for the entire harness. Survives restarts, tracks everything.

```sql
-- Every episode run
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,       -- 'exp1_scientist', 'exp2_worldmodel', 'exp3_probe_solve'
    game_id TEXT NOT NULL,          -- 'ls20', 'ft09', 'vc33', etc.
    level INTEGER,                  -- game level reached
    started_at TEXT NOT NULL,       -- ISO8601 timestamp
    finished_at TEXT,
    won BOOLEAN,
    actions_taken INTEGER,
    rhae_score REAL,               -- computed RHAE for this episode
    status TEXT DEFAULT 'running',  -- 'running', 'won', 'lost', 'crashed', 'timeout'
    error_message TEXT,
    config_json TEXT,              -- snapshot of config used for this run
    result_file TEXT               -- path to detailed JSONL
);

-- Per-experiment aggregate metrics (updated after each run)
CREATE TABLE experiment_stats (
    experiment TEXT NOT NULL,
    game_id TEXT NOT NULL,
    total_runs INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    best_rhae REAL DEFAULT 0.0,
    avg_actions_to_win REAL,
    last_run_at TEXT,
    PRIMARY KEY (experiment, game_id)
);

-- Notification log (enforces max 3 texts/day)
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at TEXT NOT NULL,          -- ISO8601
    message TEXT NOT NULL,
    milestone_type TEXT             -- 'level_solved', 'new_best_rhae', 'experiment_milestone'
);

-- Fast-fail tracking
CREATE TABLE fast_fail_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    check_name TEXT NOT NULL,
    threshold TEXT NOT NULL,
    actual_value TEXT,
    passed BOOLEAN,
    checked_at TEXT NOT NULL
);

-- Checkpoints
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    game_id TEXT,
    checkpoint_path TEXT NOT NULL,
    metric_value REAL,
    created_at TEXT NOT NULL
);
```

## Round-Robin Orchestrator (`harness.py`)

The main loop. Runs continuously. Round-robins through experiments and games.

```
┌──────────────────────────────────────────────────────────────┐
│                    HARNESS MAIN LOOP                          │
│                                                               │
│  games = [ls20, ft09, vc33] + any API-key games               │
│  experiments = [exp1, exp2, exp3]                              │
│                                                               │
│  while True:                                                  │
│    for exp in experiments:                                     │
│      if exp is fast-fail killed:                              │
│        skip                                                   │
│      for game in games:                                       │
│        1. Load exp config + any checkpoint                    │
│        2. Run one episode (with timeout)                      │
│        3. Log result to SQLite + JSONL                        │
│        4. Print progress to stdout                            │
│        5. Save checkpoint if new best                         │
│        6. Check fast-fail criteria                            │
│        7. Check milestone → maybe send iMessage               │
│        8. Release GPU memory (torch.cuda.empty_cache)         │
│                                                               │
│  Between full cycles:                                         │
│    - Print cycle summary table to stdout                      │
│    - Check if all experiments are killed → exit               │
│    - Increment cycle counter                                  │
└──────────────────────────────────────────────────────────────┘
```

### GPU Time-Sharing Strategy

The three experiments have very different GPU profiles:

| Experiment | GPU need | Sharing strategy |
|-----------|---------|-----------------|
| exp1 (LLM Scientist) | 20-24GB (vLLM server) | Runs as persistent background process. Other exps use leftover VRAM. |
| exp2 (World Model) | <1GB inference, ~2GB training | Loads model on demand, frees after episode. |
| exp3 (Probe-Solve) | <500MB | Always fits alongside anything. |

**Approach**: Start vLLM server as a daemon for exp1. Exp2 and exp3 load small models on demand. When exp2 needs to train (periodic retraining), pause the vLLM server, train, restart vLLM. This is handled by harness.py.

Alternative if vLLM is too greedy: use `llama.cpp` with `--n-gpu-layers` to control exact VRAM usage, or use the Claude API for exp1 and give the full GPU to exp2/exp3.

### Stdout Progress Format

Every episode prints one structured line:

```
[2026-03-27 14:32:05] exp1_scientist | ls20 | ep=47 | WON level=1 actions=23 rhae=0.42 | best=0.42 | cycle=3
[2026-03-27 14:32:18] exp1_scientist | ft09 | ep=48 | LOST actions=80 | best=0.00 | cycle=3
[2026-03-27 14:33:01] exp2_worldmodel | ls20 | ep=12 | LOST actions=100 (timeout) | best=0.00 | cycle=3
[2026-03-27 14:33:05] exp3_probe     | ls20 | ep=31 | WON level=1 actions=35 rhae=0.31 | best=0.31 | cycle=3
```

Every full cycle prints a summary table:

```
╔════════════════════╦════════╦══════╦══════════╦══════════╦════════════╗
║ Experiment         ║ Game   ║ Runs ║ Win Rate ║ Best RHAE║ Status     ║
╠════════════════════╬════════╬══════╬══════════╬══════════╬════════════╣
║ exp1_scientist     ║ ls20   ║  47  ║  12.8%   ║  0.42    ║ active     ║
║ exp1_scientist     ║ ft09   ║  48  ║   0.0%   ║  0.00    ║ active     ║
║ exp1_scientist     ║ vc33   ║  46  ║   2.2%   ║  0.18    ║ active     ║
║ exp2_worldmodel    ║ ls20   ║  12  ║   0.0%   ║  0.00    ║ training   ║
║ exp2_worldmodel    ║ ft09   ║  12  ║   0.0%   ║  0.00    ║ training   ║
║ exp2_worldmodel    ║ vc33   ║  12  ║   0.0%   ║  0.00    ║ training   ║
║ exp3_probe         ║ ls20   ║  31  ║   6.5%   ║  0.31    ║ active     ║
║ exp3_probe         ║ ft09   ║  31  ║   0.0%   ║  0.00    ║ active     ║
║ exp3_probe         ║ vc33   ║  31  ║   3.2%   ║  0.12    ║ active     ║
╚════════════════════╩════════╩══════╩══════════╩══════════╩════════════╝
Cycle 3 complete. Elapsed: 2h 14m. Next cycle starting...
```

## iMessage Notifications (`notify.py`)

Sends milestone alerts to Mark via BlueBubbles API on `reasonable-excuse`.

### Rules

1. **Max 3 texts per calendar day** — enforced by querying SQLite `notifications` table
2. **Only major milestones**:
   - First time any experiment wins any level of any game
   - New best RHAE score (>0.1 improvement over previous best)
   - An experiment passes all fast-fail checks (confirmed viable)
   - An experiment fails all fast-fail checks (confirmed dead)
3. **Never send between 10pm-8am MST** — queue for morning delivery

### Implementation

```python
# notify.py
import subprocess, sqlite3, datetime

DB_PATH = "arc_agi_3.db"
PHONE = "+14802822064"

def send_imessage(message: str, milestone_type: str) -> bool:
    """Send iMessage via BlueBubbles on reasonable-excuse. Returns True if sent."""
    db = sqlite3.connect(DB_PATH)

    # Check daily limit
    today = datetime.date.today().isoformat()
    count = db.execute(
        "SELECT COUNT(*) FROM notifications WHERE sent_at LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    if count >= 3:
        print(f"[notify] Suppressed (3/3 texts today): {message}")
        return False

    # Check quiet hours (10pm-8am MST = UTC-7)
    now = datetime.datetime.now()
    if now.hour >= 22 or now.hour < 8:
        print(f"[notify] Suppressed (quiet hours): {message}")
        return False

    # Send via BlueBubbles
    import json
    payload = json.dumps({
        "chatGuid": f"iMessage;-;{PHONE}",
        "message": message,
        "method": "private-api"
    })
    cmd = (
        f'ssh xeb@reasonable-excuse \'curl -s -X POST '
        f'"http://localhost:1235/api/v1/message/text?password=IhopeIgetajob1%21" '
        f'-H "Content-Type: application/json" '
        f"-d '{payload}'\""
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=15)
        if result.returncode == 0:
            db.execute(
                "INSERT INTO notifications (sent_at, message, milestone_type) VALUES (?, ?, ?)",
                (datetime.datetime.now().isoformat(), message, milestone_type)
            )
            db.commit()
            print(f"[notify] SENT: {message}")
            return True
    except Exception as e:
        print(f"[notify] Failed: {e}")
    return False


def check_and_notify(db_path: str, experiment: str, game_id: str, won: bool,
                     level: int, rhae: float):
    """Called after each episode. Decides whether to send a notification."""
    db = sqlite3.connect(db_path)

    if won:
        # Check: is this the first win for this experiment+game?
        prev_wins = db.execute(
            "SELECT COUNT(*) FROM runs WHERE experiment=? AND game_id=? AND won=1",
            (experiment, game_id)
        ).fetchone()[0]
        if prev_wins <= 1:  # this is the first (just inserted)
            send_imessage(
                f"🏆 ARC-AGI-3: {experiment} just SOLVED {game_id} level {level}! "
                f"RHAE={rhae:.2f}, first win ever for this experiment+game.",
                "level_solved"
            )
            return

    # Check: new best RHAE (significant improvement)?
    prev_best = db.execute(
        "SELECT best_rhae FROM experiment_stats WHERE experiment=? AND game_id=?",
        (experiment, game_id)
    ).fetchone()
    if prev_best and rhae > 0 and rhae - prev_best[0] > 0.1:
        send_imessage(
            f"📈 ARC-AGI-3: {experiment} on {game_id} new best RHAE={rhae:.2f} "
            f"(was {prev_best[0]:.2f}). Significant improvement.",
            "new_best_rhae"
        )
```

## Checkpointing Strategy

Each experiment saves its own state, but the harness manages checkpoint lifecycle.

| Experiment | What's checkpointed | When | Format |
|-----------|-------------------|------|--------|
| exp1_scientist | Best system prompt variant, belief state template | After each improvement in win rate | JSON |
| exp2_worldmodel | Segmenter weights, transition predictor weights, collected data | After training phase, after online updates | `.pt` + `.npz` |
| exp3_probe_solve | State encoder weights, serialized causal memory | After each game (memory grows), after encoder fine-tune | `.pt` + `.pkl` |

Checkpoints are registered in SQLite so the harness knows what to load on restart.

## Fast-Fail Evaluation

After each full cycle, the harness checks fast-fail criteria per experiment:

```python
FAST_FAIL_RULES = {
    "exp1_scientist": [
        ("json_parse_rate", "> 0.80", "after 10 episodes",
         "LLM must produce valid JSON >80% of the time"),
        ("hypothesis_convergence", "< 15 steps", "after 10 episodes",
         "Hypotheses must reach confidence >0.7 within 15 probe steps"),
        ("beats_random", "> 0.0", "after 50 episodes",
         "Must have higher win rate than random agent"),
        ("solves_any_level", ">= 1 win", "after 100 episodes",
         "Must solve at least level 1 of at least 1 game"),
    ],
    "exp2_worldmodel": [
        ("segmentation_iou", "> 0.95", "after training",
         "Object segmentation must match connected components at >95% IOU"),
        ("transition_accuracy", "> 0.85", "after training",
         "Object-level transition prediction accuracy must exceed 85%"),
        ("exploration_rate", "> random", "after 20 episodes",
         "Info-gain actions must explore faster than random"),
        ("solves_any_level", ">= 1 win", "after 100 episodes",
         "Must solve at least level 1 of at least 1 game"),
    ],
    "exp3_probe_solve": [
        ("distinct_effects", ">= 3 actions", "after 1 run per game",
         "Probes must discover distinct effects for at least 3 different actions"),
        ("memory_precision", "> 0.50", "after 3 games",
         "Cross-game memory retrieval must have >50% precision"),
        ("beats_random", "> 0.0", "after 50 episodes",
         "Must have higher win rate than random agent"),
        ("action_efficiency", "< 2x random", "after 100 episodes",
         "Total actions (probe+solve) must be < 2x random's winning count"),
    ],
}
```

When an experiment fails all criteria for a given check threshold (e.g., "after 100 episodes" and still 0 wins), it gets **killed** — the harness skips it in future cycles and sends a notification:

```
❌ ARC-AGI-3: exp2_worldmodel KILLED — failed fast-fail check
"solves_any_level" after 100 episodes (0 wins across all games).
```

## Continuous Running

### Startup

```bash
# From project root
uv run harness.py
```

The harness:
1. Creates/opens `arc_agi_3.db`, runs migrations if needed
2. Checks for existing checkpoints (resume from where we left off)
3. Starts vLLM server if exp1 is active
4. Enters the round-robin loop
5. Handles SIGTERM/SIGINT gracefully (finish current episode, checkpoint, exit)

### Restart/Resume

On restart, harness.py reads SQLite to determine:
- Which experiments are still active (not fast-fail killed)
- Which cycle we're on
- What checkpoints to load
- How many notifications have been sent today

No manual state management needed.

### Process Management

For long-running unattended execution:

```bash
# Run in tmux (recommended)
tmux new-session -d -s arc 'cd /media/xeb/GreyArea/projects/arc-agi-3 && uv run harness.py 2>&1 | tee harness.log'

# Or with nohup
nohup uv run harness.py > harness.log 2>&1 &
```

### Episode Timeout

Each episode has a hard timeout to prevent hangs:

| Experiment | Timeout per episode | Reason |
|-----------|-------------------|--------|
| exp1_scientist | 5 min | LLM inference is slow; 80 steps * ~3s/step |
| exp2_worldmodel | 2 min | Fast inference but planning can loop |
| exp3_probe_solve | 1 min | Probe battery is fixed size, solve is fast |

Timeout = `status='timeout'` in SQLite, episode counts toward fast-fail checks.

## Result Files

Each experiment writes detailed JSONL to its `results/` folder. One line per episode:

```json
{
  "timestamp": "2026-03-27T14:32:05",
  "game_id": "ls20",
  "episode": 47,
  "level_reached": 1,
  "won": true,
  "actions_taken": 23,
  "probe_actions": 12,
  "solve_actions": 11,
  "rhae_score": 0.42,
  "wall_time_s": 34.2,
  "config": {"model": "Qwen2.5-32B-AWQ", "temperature": 0.3, "probe_budget": 30},
  "hypothesis_log": ["ACTION1 moves objects right (0.9)", "goal: clear blue cells (0.7)"],
  "error": null
}
```

## Lessons from autoresearch Applied Here

| autoresearch pattern | ARC-AGI-3 adaptation |
|---------------------|---------------------|
| Single metric (val_bpb) | RHAE score per game, win rate as secondary |
| results.tsv (keep/discard/crash) | SQLite `runs` table with status column |
| git branch per experiment | Folder per experiment (simpler, no merge conflicts) |
| Fixed 5-min time budget | Per-experiment episode timeouts |
| Only modify train.py | Each experiment's `config.py` is the tuning surface |
| Never stop looping | `while True` round-robin with graceful shutdown |
| Agent reads current state before modifying | Harness loads checkpoints + SQLite state on resume |

## Dependencies

All managed via `uv add --script`:

```bash
# Core harness
uv add --script='harness.py' 'arc-agi' 'numpy' 'torch'

# Experiment 1 (LLM Scientist)
uv add --script='exp1_scientist/agent.py' 'arc-agi' 'openai' 'numpy'

# Experiment 2 (World Model)
uv add --script='exp2_worldmodel/train.py' 'torch' 'numpy'
uv add --script='exp2_worldmodel/agent.py' 'torch' 'arc-agi' 'numpy'

# Experiment 3 (Probe-Solve)
uv add --script='exp3_probe_solve/agent.py' 'torch' 'arc-agi' 'numpy'

# Notification
uv add --script='notify.py' ''  # stdlib only, no extra deps

# vLLM for local LLM serving (exp1)
# Installed separately as a persistent service:
# uv tool install vllm
```
