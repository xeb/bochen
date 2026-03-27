# Bochen (בֹּחֵן)

*"One who tests and examines"* — from the Hebrew root used in Proverbs 17:3 for testing silver.

Mark's personal project to burn an RTX 4090 exploring ways to solve [ARC-AGI-3](https://arcprize.org/arc-agi/3/), the interactive reasoning benchmark that measures an AI agent's ability to generalize in novel, unseen environments.

## What is ARC-AGI-3?

ARC-AGI-3 is a benchmark of turn-based 2D grid puzzles where agents must discover hidden rules through interaction — not pattern matching on static examples. It measures five capabilities: Exploration, Percept-Plan-Action cycles, Memory, Goal Acquisition, and Alignment. Agents are scored by Relative Human Action Efficiency (RHAE): how many actions they take to solve puzzles compared to humans.

## The Approach

After evaluating 10 GPU-accelerated ideas and a detailed counter-analysis, three experiments survived:

### Experiment 1: LLM Scientist Agent (40-50% estimated probability)

A local open-weight LLM (Qwen2.5-32B-AWQ) running on the RTX 4090 via vLLM acts as an experimental scientist. It observes structured grid state, proposes hypotheses about rules and goals, designs discriminating experiments, executes them, and prunes disproven hypotheses. Zero API cost means thousands of iterations for prompt and loop tuning.

### Experiment 2: Object-Centric World Model + Information-Gain Search (30-40%)

A GPU-trained CNN segments grids into objects and learns to predict object-level transitions (moved, deleted, color changed, merged). Search picks actions that maximize information gain about game mechanics, not just actions that change pixels. Deterministic games mean the model can theoretically be perfect.

### Experiment 3: Two-Phase Probe-Solve with Causal Memory (30-38%)

Hard split between probing (systematic diagnostic battery: try each action, click each object, test undo, check consistency) and solving (retrieve relevant causal memories, plan, execute). GPU-accelerated memory embedding and retrieval enables cross-game transfer.

## Infrastructure

- **Round-robin harness**: Continuously cycles through all 3 experiments across all available games. Fast-fail criteria automatically kill experiments that don't show signal.
- **SQLite state database**: Tracks every run, experiment stats, checkpoints, and notification history. Survives restarts for continuous unattended operation.
- **GPU time-sharing**: vLLM daemon for the LLM agent (20-24GB), small models loaded on demand for world model and probe-solve (<1GB each).
- **Checkpointing**: Each experiment saves its own state (prompts, model weights, causal memories) for resume after restart.
- **iMessage alerts**: Milestone notifications (first win, significant RHAE improvement, experiment death) via BlueBubbles, max 3/day.

## Hardware

- NVIDIA GeForce RTX 4090 (24GB VRAM)
- PyTorch 2.8.0+cu128, CUDA 12.4
- Local LLM serving via vLLM

## Running

```bash
uv run harness.py
```

## Project Structure

```
arc-agi-3/
├── harness.py              # Main round-robin orchestrator
├── db.py                   # SQLite state manager
├── notify.py               # iMessage milestone alerts
├── shared/                 # Grid perception, metrics, env wrapper
├── exp1_scientist/         # LLM hypothesis-driven agent
├── exp2_worldmodel/        # Object-centric world model + info-gain search
└── exp3_probe_solve/       # Two-phase probe-solve with causal memory
```

## Docs

- `three.md` — Full spec for the three experiments and infrastructure
- `CLAUDE.md` — ARC-AGI-3 API reference and project instructions
- `ten.md` — Original 10 ideas (historical)
- `COUNTER.md` — Critique that shaped the final three
