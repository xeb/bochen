#!/usr/bin/env python3
"""
Bochen Research Harness — continuous round-robin orchestrator for ARC-AGI-3 experiments.

Usage:
    uv run harness.py [--games ls20,ft09,vc33] [--experiments exp1,exp2,exp3]
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from db import BochenDB
from notify import check_and_notify, notify_experiment_killed
from shared.metrics import format_result_line, format_summary_table, result_to_jsonl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_GAMES = ["ls20", "ft09", "vc33"]

EXPERIMENT_TIMEOUTS = {
    "exp1_scientist": 600,      # 10 min (LLM inference is slow)
    "exp2_worldmodel": 120,     # 2 min
    "exp3_probe_solve": 60,     # 1 min
}

FAST_FAIL_RULES = {
    "exp1_scientist": [
        {"name": "solves_any_level", "after_episodes": 100,
         "check": lambda db, exp: db.get_win_count(exp) >= 1,
         "description": "Must solve at least 1 level after 100 episodes"},
    ],
    "exp2_worldmodel": [
        {"name": "solves_any_level", "after_episodes": 100,
         "check": lambda db, exp: db.get_win_count(exp) >= 1,
         "description": "Must solve at least 1 level after 100 episodes"},
    ],
    "exp3_probe_solve": [
        {"name": "solves_any_level", "after_episodes": 100,
         "check": lambda db, exp: db.get_win_count(exp) >= 1,
         "description": "Must solve at least 1 level after 100 episodes"},
    ],
}

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

# Lazy-loaded agents to avoid importing everything at startup
_agents = {}


def get_agent(experiment: str):
    """Lazy-load and cache experiment agents."""
    if experiment not in _agents:
        if experiment == "exp1_scientist":
            from exp1_scientist.agent import ScientistAgent
            _agents[experiment] = ScientistAgent()
        elif experiment == "exp2_worldmodel":
            from exp2_worldmodel.agent import WorldModelAgent
            ckpt = PROJECT_ROOT / "exp2_worldmodel" / "checkpoints" / "world_model_best.pt"
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


def run_single_episode(experiment: str, game_id: str) -> dict:
    """Run one episode of an experiment on a game. Returns result dict."""
    agent = get_agent(experiment)
    result = agent.run_episode(game_id)
    result["experiment"] = experiment
    return result


# ---------------------------------------------------------------------------
# Fast-fail evaluation
# ---------------------------------------------------------------------------

def check_fast_fail(db: BochenDB, experiment: str) -> bool:
    """Check fast-fail criteria. Returns True if experiment should be killed."""
    total_episodes = db.get_episode_count(experiment)
    rules = FAST_FAIL_RULES.get(experiment, [])

    for rule in rules:
        if total_episodes >= rule["after_episodes"]:
            passed = rule["check"](db, experiment)
            db.log_fast_fail(
                experiment, rule["name"],
                f"after {rule['after_episodes']} episodes",
                str(passed), passed
            )
            if not passed:
                print(f"\n  [FAST-FAIL] {experiment}: {rule['description']}")
                return True
    return False


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def log_result(db: BochenDB, experiment: str, game_id: str, episode: int,
               result: dict, cycle: int):
    """Log result to SQLite, JSONL file, and stdout."""
    won = result.get("won", False)
    actions = result.get("actions", 0)
    level = result.get("level", 0)
    rhae = result.get("rhae_score", 0.0)

    # Write to JSONL
    results_dir = PROJECT_ROOT / experiment / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    jsonl_path = results_dir / f"{date_str}.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(result_to_jsonl(result) + "\n")

    # Update SQLite
    db.update_stats(experiment, game_id)
    best_rhae = db.get_best_rhae(experiment, game_id)

    # Print progress
    line = format_result_line(
        experiment, game_id, episode, won, level, actions,
        rhae, best_rhae, cycle
    )
    print(line)

    # Check notifications
    check_and_notify(db, experiment, game_id, won, level, rhae)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoints(experiment: str, db: BochenDB):
    """Save experiment-specific checkpoints."""
    agent = _agents.get(experiment)
    if agent is None:
        return

    if experiment == "exp3_probe_solve":
        mem_path = str(PROJECT_ROOT / "exp3_probe_solve" / "checkpoints" / "memory.pkl")
        agent.save_memory(mem_path)
        db.save_checkpoint(experiment, None, mem_path, agent.memory.size)


# ---------------------------------------------------------------------------
# World model training (exp2 bootstrap)
# ---------------------------------------------------------------------------

def maybe_train_world_model(db: BochenDB, games: list[str]):
    """Check if world model needs initial training, and do it if so."""
    ckpt_path = PROJECT_ROOT / "exp2_worldmodel" / "checkpoints" / "world_model_best.pt"
    if ckpt_path.exists():
        return  # already trained

    # Check if we have collected data
    data_dir = PROJECT_ROOT / "exp2_worldmodel" / "data"
    data_files = list(data_dir.glob("*_transitions.npz")) if data_dir.exists() else []

    if not data_files:
        print("\n[exp2] Collecting transition data for world model training...")
        from exp2_worldmodel.collect import collect_transitions
        for game_id in games:
            # Small bootstrap set — fast start, retrain later with more data
            collect_transitions(game_id, num_steps=2000,
                              save_dir=str(data_dir))
        data_files = list(data_dir.glob("*_transitions.npz"))

    if data_files:
        print("\n[exp2] Training world model...")
        from exp2_worldmodel.train import train_world_model
        metrics = train_world_model(
            [str(f) for f in data_files],
            save_dir=str(PROJECT_ROOT / "exp2_worldmodel" / "checkpoints")
        )
        print(f"[exp2] Training complete: {metrics}")
        # Reload agent with new checkpoint
        if "exp2_worldmodel" in _agents:
            del _agents["exp2_worldmodel"]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_shutdown = False


def handle_signal(signum, frame):
    global _shutdown
    print("\n\n[harness] Shutdown signal received. Finishing current episode...")
    _shutdown = True


def main():
    global _shutdown

    parser = argparse.ArgumentParser(description="Bochen Research Harness")
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES),
                        help="Comma-separated game IDs")
    parser.add_argument("--experiments", default="exp1_scientist,exp2_worldmodel,exp3_probe_solve",
                        help="Comma-separated experiment names")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",")]
    experiments = [e.strip() for e in args.experiments.split(",")]

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    db = BochenDB()
    killed_experiments = set()

    print("=" * 80)
    print("  BOCHEN - ARC-AGI-3 Research Harness")
    print(f"  Games: {games}")
    print(f"  Experiments: {experiments}")
    print(f"  Started: {datetime.datetime.now().isoformat()}")
    print("=" * 80)

    # Check which experiments are already killed
    for exp in experiments:
        if db.is_experiment_killed(exp):
            killed_experiments.add(exp)
            print(f"  [skip] {exp} previously killed by fast-fail")

    # Bootstrap world model if needed
    if "exp2_worldmodel" in experiments and "exp2_worldmodel" not in killed_experiments:
        try:
            maybe_train_world_model(db, games)
        except Exception as e:
            print(f"  [exp2] Training failed: {e}. Will use heuristic agent.")

    cycle = 0
    while not _shutdown:
        cycle += 1
        active_experiments = [e for e in experiments if e not in killed_experiments]

        if not active_experiments:
            print("\n[harness] All experiments killed. Exiting.")
            break

        print(f"\n{'='*80}")
        print(f"  Cycle {cycle} | Active: {active_experiments}")
        print(f"{'='*80}")

        for experiment in active_experiments:
            if _shutdown:
                break

            for game_id in games:
                if _shutdown:
                    break

                episode = db.get_episode_count(experiment, game_id) + 1
                timeout = EXPERIMENT_TIMEOUTS.get(experiment, 120)

                # Start run
                run_id = db.start_run(experiment, game_id, episode)

                try:
                    # Run with timeout using thread pool
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_single_episode, experiment, game_id)
                        try:
                            result = future.result(timeout=timeout)
                        except FuturesTimeout:
                            result = {
                                "won": False, "level": 0, "actions": 0,
                                "game_id": game_id, "experiment": experiment,
                            }
                            db.finish_run(run_id, False, 0, status="timeout")
                            print(f"  [timeout] {experiment} on {game_id} after {timeout}s")
                            continue

                    # Log result
                    won = result.get("won", False)
                    actions = result.get("actions", 0)
                    level = result.get("level", 0)
                    rhae = result.get("rhae_score", 0.0)
                    status = "won" if won else "lost"

                    db.finish_run(run_id, won, actions, level, rhae, status,
                                 result_file=str(PROJECT_ROOT / experiment / "results"))
                    log_result(db, experiment, game_id, episode, result, cycle)

                except Exception as e:
                    tb = traceback.format_exc()
                    db.finish_run(run_id, False, 0, status="crashed",
                                 error=str(e))
                    print(f"  [crash] {experiment} on {game_id}: {e}")
                    print(f"  {tb[:200]}")

                # Free GPU memory between experiments
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save checkpoints after each full game sweep
            try:
                save_checkpoints(experiment, db)
            except Exception as e:
                print(f"  [checkpoint] {experiment} save failed: {e}")

            # Check fast-fail after each experiment's game sweep
            if check_fast_fail(db, experiment):
                killed_experiments.add(experiment)
                db.set_experiment_status(experiment, "killed")
                notify_experiment_killed(
                    db, experiment,
                    f"Failed fast-fail after {db.get_episode_count(experiment)} episodes"
                )

        # Cycle summary
        all_stats = db.get_stats()
        for stat in all_stats:
            if stat["experiment"] in killed_experiments:
                stat["status"] = "killed"
        if all_stats:
            print(f"\n{format_summary_table(all_stats)}")
            print(f"Cycle {cycle} complete. "
                  f"Active: {len(active_experiments)}/{len(experiments)} experiments. "
                  f"Killed: {killed_experiments or 'none'}")

    # Graceful shutdown
    print("\n[harness] Saving final checkpoints...")
    for exp in experiments:
        if exp not in killed_experiments:
            try:
                save_checkpoints(exp, db)
            except Exception:
                pass
    db.close()
    print("[harness] Shutdown complete.")


if __name__ == "__main__":
    main()
