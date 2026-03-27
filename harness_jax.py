#!/usr/bin/env python3
"""JAX harness — runs exp2 and exp3 using JAX models alongside the torch harness.

Usage:
    .venv/bin/python harness_jax.py [--games ls20,ft09,vc33] [--experiments exp2_worldmodel,exp3_probe_solve]
"""

import sys
import os
import signal
import time
import json
import argparse
import datetime
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Default config
DEFAULT_GAMES = ["ls20", "ft09", "vc33"]
DEFAULT_EXPERIMENTS = ["exp2_worldmodel", "exp3_probe_solve"]

# ---------------------------------------------------------------------------
# Lazy-loaded JAX agents
# ---------------------------------------------------------------------------
_agents = {}


def get_agent(experiment: str):
    if experiment not in _agents:
        if experiment == "exp2_worldmodel":
            from jax_models.exp2_worldmodel.agent import WorldModelAgent
            ckpt = PROJECT_ROOT / "exp2_worldmodel" / "checkpoints" / "world_model_best_jax"
            _agents[experiment] = WorldModelAgent(
                checkpoint_path=str(ckpt) if ckpt.exists() else None
            )
        elif experiment == "exp3_probe_solve":
            from jax_models.exp3_probe_solve.agent import ProbeSolveAgent
            mem_path = PROJECT_ROOT / "jax_models" / "exp3_probe_solve" / "checkpoints" / "memory.pkl"
            _agents[experiment] = ProbeSolveAgent(
                memory_path=str(mem_path) if mem_path.exists() else None
            )
    return _agents[experiment]


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "jax_models" / "results"


def log_result(experiment, game_id, episode, result, cycle):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = RESULTS_DIR / f"{today}.jsonl"
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "experiment": experiment,
        "game_id": game_id,
        "episode": episode,
        "cycle": cycle,
        "backend": "jax",
        **result,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    won = result.get("won", False)
    actions = result.get("actions", 0)
    status = "WON" if won else "LOST"
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"jax/{experiment:<20s} | {game_id:<6s} | ep={episode:<5d} | "
          f"{status} actions={actions}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[jax harness] Shutting down gracefully...")


def main():
    global _shutdown

    parser = argparse.ArgumentParser(description="JAX experiment harness")
    parser.add_argument("--games", type=str, default=",".join(DEFAULT_GAMES))
    parser.add_argument("--experiments", type=str, default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--max-cycles", type=int, default=0, help="0 = unlimited")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",")]
    experiments = [e.strip() for e in args.experiments.split(",")]

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[jax harness] Starting: experiments={experiments}, games={games}")
    print(f"[jax harness] Results -> {RESULTS_DIR}/")

    cycle = 0
    episode_counts = {e: 0 for e in experiments}

    while not _shutdown:
        cycle += 1
        if args.max_cycles and cycle > args.max_cycles:
            print(f"[jax harness] Reached max cycles ({args.max_cycles}), stopping.")
            break

        print(f"\n--- JAX Cycle {cycle} | Active: {experiments} ---")

        for experiment in experiments:
            if _shutdown:
                break

            for game_id in games:
                if _shutdown:
                    break

                episode_counts[experiment] += 1
                ep = episode_counts[experiment]

                try:
                    agent = get_agent(experiment)
                    result = agent.run_episode(game_id)
                    result["experiment"] = experiment
                    log_result(experiment, game_id, ep, result, cycle)

                    # Save memory for exp3
                    if experiment == "exp3_probe_solve" and hasattr(agent, "save_memory"):
                        mem_dir = PROJECT_ROOT / "jax_models" / "exp3_probe_solve" / "checkpoints"
                        mem_dir.mkdir(parents=True, exist_ok=True)
                        agent.save_memory(str(mem_dir / "memory.pkl"))

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  [crash] jax/{experiment} on {game_id}: {e}")
                    print(f"  {tb[:300]}")

    print(f"\n[jax harness] Done. Ran {sum(episode_counts.values())} total episodes.")


if __name__ == "__main__":
    main()
