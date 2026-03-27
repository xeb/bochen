"""iMessage milestone notifications via BlueBubbles on reasonable-excuse."""

import subprocess
import json
import datetime
from db import BochenDB

PHONE = "+14802822064"
MAX_PER_DAY = 3
QUIET_START = 22  # 10pm
QUIET_END = 8     # 8am


def send_imessage(message: str) -> bool:
    """Send iMessage via BlueBubbles. Returns True if sent successfully."""
    payload = json.dumps({
        "chatGuid": f"iMessage;-;{PHONE}",
        "message": message,
        "method": "private-api"
    })
    cmd = (
        f"ssh xeb@reasonable-excuse 'curl -s -X POST "
        f"\"http://localhost:1235/api/v1/message/text?password=IhopeIgetajob1%21\" "
        f"-H \"Content-Type: application/json\" "
        f"-d {repr(payload)}'"
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=15, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[notify] Send failed: {e}")
        return False


def check_and_notify(db: BochenDB, experiment: str, game_id: str,
                     won: bool, level: int, rhae: float):
    """Decide whether to send a notification based on milestone significance."""
    now = datetime.datetime.now()

    # Quiet hours
    if now.hour >= QUIET_START or now.hour < QUIET_END:
        return

    # Daily limit
    if db.notifications_today() >= MAX_PER_DAY:
        return

    message = None
    milestone_type = None

    if won:
        # First win for this experiment+game?
        prev_wins = db.get_win_count(experiment, game_id)
        if prev_wins <= 1:
            message = (
                f"Bochen: {experiment} SOLVED {game_id} level {level}! "
                f"RHAE={rhae:.2f}. First win for this experiment+game."
            )
            milestone_type = "level_solved"

    # New best RHAE (significant improvement)?
    if message is None and rhae > 0:
        prev_best = db.get_best_rhae(experiment, game_id)
        if rhae - prev_best > 0.1:
            message = (
                f"Bochen: {experiment} on {game_id} new best RHAE={rhae:.2f} "
                f"(was {prev_best:.2f})."
            )
            milestone_type = "new_best_rhae"

    if message:
        if send_imessage(message):
            db.log_notification(message, milestone_type)
            print(f"[notify] SENT: {message}")
        else:
            print(f"[notify] Failed to send: {message}")


def notify_experiment_killed(db: BochenDB, experiment: str, reason: str):
    """Notify that an experiment has been killed."""
    now = datetime.datetime.now()
    if now.hour >= QUIET_START or now.hour < QUIET_END:
        return
    if db.notifications_today() >= MAX_PER_DAY:
        return

    message = f"Bochen: {experiment} KILLED - {reason}"
    if send_imessage(message):
        db.log_notification(message, "experiment_killed")
        print(f"[notify] SENT: {message}")
