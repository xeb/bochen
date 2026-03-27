"""RHAE scoring and result formatting."""

import json
import datetime


def compute_rhae(human_baseline_actions: int, ai_actions: int) -> float:
    """Compute Relative Human Action Efficiency score for a single level."""
    if ai_actions <= 0:
        return 0.0
    raw = (human_baseline_actions / ai_actions) ** 2
    return min(1.0, raw)


def format_result_line(experiment: str, game_id: str, episode: int,
                       won: bool, level: int, actions: int,
                       rhae: float, best_rhae: float, cycle: int,
                       extra: str = "") -> str:
    """Format a single-line progress string for stdout."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = f"WON level={level} actions={actions} rhae={rhae:.2f}" if won else f"LOST actions={actions}"
    line = f"[{ts}] {experiment:20s} | {game_id:6s} | ep={episode:<4d} | {status} | best={best_rhae:.2f} | cycle={cycle}"
    if extra:
        line += f" | {extra}"
    return line


def format_summary_table(stats: list[dict]) -> str:
    """Format a cycle summary table from experiment_stats rows."""
    header = f"{'Experiment':20s} | {'Game':6s} | {'Runs':>5s} | {'Win%':>6s} | {'BestRHAE':>8s} | {'Status':10s}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row in stats:
        win_pct = f"{100 * row['total_wins'] / max(1, row['total_runs']):.1f}%"
        lines.append(
            f"{row['experiment']:20s} | {row['game_id']:6s} | {row['total_runs']:5d} | "
            f"{win_pct:>6s} | {row['best_rhae']:8.2f} | {row['status']:10s}"
        )
    lines.append(sep)
    return "\n".join(lines)


def result_to_jsonl(data: dict) -> str:
    """Serialize a result dict to a single JSONL line."""
    data["timestamp"] = datetime.datetime.now().isoformat()
    return json.dumps(data, default=str)
