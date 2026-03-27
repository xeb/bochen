"""SQLite state database for the Bochen research harness."""

import sqlite3
import datetime
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "arc_agi_3.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    game_id TEXT NOT NULL,
    episode INTEGER,
    level INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    won BOOLEAN DEFAULT 0,
    actions_taken INTEGER DEFAULT 0,
    rhae_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'running',
    error_message TEXT,
    config_json TEXT,
    result_file TEXT
);

CREATE TABLE IF NOT EXISTS experiment_stats (
    experiment TEXT NOT NULL,
    game_id TEXT NOT NULL,
    total_runs INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    best_rhae REAL DEFAULT 0.0,
    avg_actions_to_win REAL DEFAULT 0.0,
    last_run_at TEXT,
    status TEXT DEFAULT 'active',
    PRIMARY KEY (experiment, game_id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at TEXT NOT NULL,
    message TEXT NOT NULL,
    milestone_type TEXT
);

CREATE TABLE IF NOT EXISTS fast_fail_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    check_name TEXT NOT NULL,
    threshold TEXT NOT NULL,
    actual_value TEXT,
    passed BOOLEAN,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    game_id TEXT,
    checkpoint_path TEXT NOT NULL,
    metric_value REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_exp_game ON runs(experiment, game_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_notifications_date ON notifications(sent_at);
"""


class BochenDB:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def start_run(self, experiment: str, game_id: str, episode: int,
                  config: dict = None) -> int:
        """Record start of a run. Returns run ID."""
        cur = self.conn.execute(
            "INSERT INTO runs (experiment, game_id, episode, started_at, config_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (experiment, game_id, episode,
             datetime.datetime.now().isoformat(),
             json.dumps(config) if config else None)
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, won: bool, actions: int, level: int = 0,
                   rhae: float = 0.0, status: str = None, error: str = None,
                   result_file: str = None):
        """Record completion of a run."""
        if status is None:
            status = "won" if won else "lost"
        self.conn.execute(
            "UPDATE runs SET finished_at=?, won=?, actions_taken=?, level=?, "
            "rhae_score=?, status=?, error_message=?, result_file=? WHERE id=?",
            (datetime.datetime.now().isoformat(), won, actions, level,
             rhae, status, error, result_file, run_id)
        )
        self.conn.commit()

    def update_stats(self, experiment: str, game_id: str):
        """Recompute experiment_stats from runs table."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(won) as wins, MAX(rhae_score) as best, "
            "AVG(CASE WHEN won=1 THEN actions_taken END) as avg_win_actions "
            "FROM runs WHERE experiment=? AND game_id=? AND status != 'running'",
            (experiment, game_id)
        ).fetchone()

        self.conn.execute(
            "INSERT INTO experiment_stats (experiment, game_id, total_runs, total_wins, "
            "best_rhae, avg_actions_to_win, last_run_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(experiment, game_id) DO UPDATE SET "
            "total_runs=?, total_wins=?, best_rhae=?, avg_actions_to_win=?, last_run_at=?",
            (experiment, game_id, row["total"], row["wins"] or 0,
             row["best"] or 0.0, row["avg_win_actions"] or 0.0,
             datetime.datetime.now().isoformat(),
             row["total"], row["wins"] or 0,
             row["best"] or 0.0, row["avg_win_actions"] or 0.0,
             datetime.datetime.now().isoformat())
        )
        self.conn.commit()

    def get_stats(self, experiment: str = None) -> list[dict]:
        """Get experiment_stats rows."""
        if experiment:
            rows = self.conn.execute(
                "SELECT *, 'active' as status FROM experiment_stats WHERE experiment=?",
                (experiment,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT *, 'active' as status FROM experiment_stats"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_episode_count(self, experiment: str, game_id: str = None) -> int:
        """Count completed episodes for an experiment."""
        if game_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment=? AND game_id=? AND status != 'running'",
                (experiment, game_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment=? AND status != 'running'",
                (experiment,)
            ).fetchone()
        return row[0]

    def get_win_count(self, experiment: str, game_id: str = None) -> int:
        """Count wins for an experiment."""
        if game_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment=? AND game_id=? AND won=1",
                (experiment, game_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment=? AND won=1",
                (experiment,)
            ).fetchone()
        return row[0]

    def get_best_rhae(self, experiment: str, game_id: str) -> float:
        """Get best RHAE for experiment+game."""
        row = self.conn.execute(
            "SELECT best_rhae FROM experiment_stats WHERE experiment=? AND game_id=?",
            (experiment, game_id)
        ).fetchone()
        return row["best_rhae"] if row else 0.0

    def notifications_today(self) -> int:
        """Count notifications sent today."""
        today = datetime.date.today().isoformat()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE sent_at LIKE ?",
            (f"{today}%",)
        ).fetchone()
        return row[0]

    def log_notification(self, message: str, milestone_type: str):
        """Record a sent notification."""
        self.conn.execute(
            "INSERT INTO notifications (sent_at, message, milestone_type) VALUES (?, ?, ?)",
            (datetime.datetime.now().isoformat(), message, milestone_type)
        )
        self.conn.commit()

    def log_fast_fail(self, experiment: str, check_name: str, threshold: str,
                      actual_value: str, passed: bool):
        """Record a fast-fail check result."""
        self.conn.execute(
            "INSERT INTO fast_fail_checks (experiment, check_name, threshold, "
            "actual_value, passed, checked_at) VALUES (?, ?, ?, ?, ?, ?)",
            (experiment, check_name, threshold, actual_value, passed,
             datetime.datetime.now().isoformat())
        )
        self.conn.commit()

    def set_experiment_status(self, experiment: str, status: str):
        """Set status for all game entries of an experiment."""
        self.conn.execute(
            "UPDATE experiment_stats SET status=? WHERE experiment=?",
            (status, experiment)
        )
        self.conn.commit()

    def is_experiment_killed(self, experiment: str) -> bool:
        """Check if experiment has been killed by fast-fail."""
        row = self.conn.execute(
            "SELECT status FROM experiment_stats WHERE experiment=? LIMIT 1",
            (experiment,)
        ).fetchone()
        return row and row["status"] == "killed" if row else False

    def save_checkpoint(self, experiment: str, game_id: str, path: str,
                        metric: float = None):
        """Register a checkpoint."""
        self.conn.execute(
            "INSERT INTO checkpoints (experiment, game_id, checkpoint_path, "
            "metric_value, created_at) VALUES (?, ?, ?, ?, ?)",
            (experiment, game_id, path, metric,
             datetime.datetime.now().isoformat())
        )
        self.conn.commit()

    def get_latest_checkpoint(self, experiment: str, game_id: str = None) -> str:
        """Get path to most recent checkpoint."""
        if game_id:
            row = self.conn.execute(
                "SELECT checkpoint_path FROM checkpoints "
                "WHERE experiment=? AND game_id=? ORDER BY created_at DESC LIMIT 1",
                (experiment, game_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT checkpoint_path FROM checkpoints "
                "WHERE experiment=? ORDER BY created_at DESC LIMIT 1",
                (experiment,)
            ).fetchone()
        return row["checkpoint_path"] if row else None

    def close(self):
        self.conn.close()
