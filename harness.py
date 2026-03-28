#!/usr/bin/env python3
"""Bochen JAX harness — continuous round-robin over all experiments.

Usage:
    .venv/bin/python harness.py [--games ls20,ft09,vc33] [--experiments imagine,exp2_worldmodel,exp3_probe_solve] [--max-cycles 0]
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
sys.path.insert(0, str(PROJECT_ROOT))

from db import BochenDB
from notify import check_and_notify, notify_experiment_killed

# Default config
DEFAULT_GAMES = ["ls20", "ft09", "vc33"]
DEFAULT_EXPERIMENTS = ["imagine", "exp2_worldmodel", "exp3_probe_solve"]

EXPERIMENT_TIMEOUTS = {
    "imagine": 300,          # probe+train+search+execute
    "exp2_worldmodel": 120,
    "exp3_probe_solve": 60,
}

# ---------------------------------------------------------------------------
# Lazy-loaded JAX agents
# ---------------------------------------------------------------------------
_agents = {}


def get_agent(experiment: str):
    if experiment not in _agents:
        if experiment == "imagine":
            from imagine_agent import ImagineAgent
            _agents[experiment] = ImagineAgent()
        elif experiment == "exp2_worldmodel":
            from exp2_worldmodel.agent import WorldModelAgent
            ckpt = PROJECT_ROOT / "exp2_worldmodel" / "checkpoints" / "world_model_best_jax"
            _agents[experiment] = WorldModelAgent(
                checkpoint_path=str(ckpt) if ckpt.exists() else None
            )
        elif experiment == "exp3_probe_solve":
            from exp3_probe_solve.agent import ProbeSolveAgent
            mem_path = PROJECT_ROOT / "exp3_probe_solve" / "checkpoints" / "memory.pkl"
            _agents[experiment] = ProbeSolveAgent(
                memory_path=str(mem_path) if mem_path.exists() else None
            )
    return _agents[experiment]


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"


def log_result(db: BochenDB, experiment: str, game_id: str, episode: int,
               result: dict, cycle: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = RESULTS_DIR / f"{today}.jsonl"
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "experiment": experiment,
        "game_id": game_id,
        "episode": episode,
        "cycle": cycle,
        **result,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    won = result.get("won", False)
    actions = result.get("actions", 0)
    level = result.get("level", 0)
    rhae = result.get("rhae_score", 0.0)
    phase = result.get("phase", "")
    status = f"WON level={level}" if won else "LOST"

    # Update DB
    db.update_stats(experiment, game_id)
    best_rhae = db.get_best_rhae(experiment, game_id)

    print(f"[{datetime.datetime.now():%H:%M:%S}] "
          f"{experiment:<20s} | {game_id:<6s} | ep={episode:<4d} | "
          f"{status} actions={actions} | {phase} | best_rhae={best_rhae:.2f} | cycle={cycle}")

    # Notifications
    check_and_notify(db, experiment, game_id, won, level, rhae)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[harness] Shutting down gracefully...")


def main():
    global _shutdown

    parser = argparse.ArgumentParser(description="Bochen JAX harness")
    parser.add_argument("--games", type=str, default=",".join(DEFAULT_GAMES))
    parser.add_argument("--experiments", type=str, default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--max-cycles", type=int, default=0, help="0 = unlimited")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",")]
    experiments = [e.strip() for e in args.experiments.split(",")]

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    db = BochenDB()
    killed = set()

    # Check previously killed experiments
    for exp in experiments:
        if db.is_experiment_killed(exp):
            killed.add(exp)

    print("=" * 70)
    print("  BOCHEN (JAX) — ARC-AGI-3 Research Harness")
    print(f"  Games:       {games}")
    print(f"  Experiments: {experiments}")
    print(f"  Killed:      {killed or 'none'}")
    print(f"  Started:     {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    cycle = 0
    episode_counts = {e: db.get_episode_count(e) for e in experiments}

    while not _shutdown:
        cycle += 1
        if args.max_cycles and cycle > args.max_cycles:
            print(f"\n[harness] Reached max cycles ({args.max_cycles}).")
            break

        active = [e for e in experiments if e not in killed]
        if not active:
            print("\n[harness] All experiments killed. Exiting.")
            break

        print(f"\n--- Cycle {cycle} | Active: {active} ---")

        for experiment in active:
            if _shutdown:
                break

            for game_id in games:
                if _shutdown:
                    break

                episode_counts[experiment] = episode_counts.get(experiment, 0) + 1
                ep = episode_counts[experiment]
                run_id = db.start_run(experiment, game_id, ep)

                try:
                    agent = get_agent(experiment)
                    result = agent.run_episode(game_id)
                    result["experiment"] = experiment

                    won = result.get("won", False)
                    actions = result.get("actions", 0)
                    level = result.get("level", 0)
                    rhae = result.get("rhae_score", 0.0)

                    db.finish_run(run_id, won, actions, level, rhae,
                                  status="won" if won else "lost")
                    log_result(db, experiment, game_id, ep, result, cycle)

                    # Save exp3 memory
                    if experiment == "exp3_probe_solve" and hasattr(agent, "save_memory"):
                        mem_dir = PROJECT_ROOT / "exp3_probe_solve" / "checkpoints"
                        mem_dir.mkdir(parents=True, exist_ok=True)
                        agent.save_memory(str(mem_dir / "memory.pkl"))

                except Exception as e:
                    tb = traceback.format_exc()
                    db.finish_run(run_id, False, 0, status="crashed", error=str(e))
                    print(f"  [crash] {experiment} on {game_id}: {e}")
                    print(f"  {tb[:300]}")

        # Fast-fail check
        for experiment in active:
            total = db.get_episode_count(experiment)
            wins = db.get_win_count(experiment)
            if total >= 100 and wins == 0:
                print(f"\n  [FAST-FAIL] {experiment}: 0 wins after {total} episodes")
                killed.add(experiment)
                db.set_experiment_status(experiment, "killed")
                notify_experiment_killed(db, experiment,
                                        f"0 wins after {total} episodes")

        # Cycle summary
        print(f"\n{'Experiment':<22s} | {'Game':<6s} | {'Runs':>5s} | {'Wins':>4s} | {'Best':>6s}")
        print("-" * 55)
        for stat in db.get_stats():
            status = " KILLED" if stat["experiment"] in killed else ""
            print(f"{stat['experiment']:<22s} | {stat['game_id']:<6s} | "
                  f"{stat['total_runs']:5d} | {stat['total_wins']:4d} | "
                  f"{stat['best_rhae']:6.2f}{status}")

    db.close()
    print("[harness] Shutdown complete.")


if __name__ == "__main__":
    main()
