# ARC-AGI-3

An Interactive Reasoning Benchmark designed to measure an AI Agent's ability to generalize in novel, unseen environments.

Measures five critical capabilities: Exploration, Percept-Plan-Action cycles, Memory, Goal Acquisition, and Alignment.

## Installation

```bash
uv add --script='main.py' 'arc-agi'
```

Or for project-level:
```bash
uv add arc-agi
```

Optionally set API key (required for non-default games):
```bash
export ARC_API_KEY=<your-key>
```
Register for a free key at arcprize.org/platform. Anonymous access gives 3 default games (ls20, ft09, vc33).

## Running Agents

Clone the agents repo first:
```bash
git clone <arc-agi-3-agents-repo>
cd arc-agi-3-agents
```

### Run a single game
```bash
uv run main.py --agent=random --game=ls20
```

### Run all games (swarm mode)
```bash
uv run main.py --agent=random
```

### Run with tags for tracking
```bash
uv run main.py --agent=llm --tags "experiment,gpt-4,baseline"
```

### Run multiple specific games
```bash
uv run main.py --agent=llm --game=ls20,ft09
```

## Built-in Agent Types

| Flag | Description | Model |
|------|-------------|-------|
| `--agent=random` | Random actions (template) | N/A |
| `--agent=llm` | OpenAI function calling, 10-msg history | gpt-4o-mini |
| `--agent=fastllm` | Skips observation step (faster, less informed) | gpt-4o-mini |
| `--agent=reasoningllm` | Captures reasoning tokens/thought process | o4-mini |
| `--agent=guidedllm` | High reasoning effort (educational only, won't generalize) | o3 |

## Creating Custom Agents

1. Copy template: `cp agents/templates/random_agent.py agents/my_agent.py`
2. Rename the class inside the file
3. Implement `is_done()` and `choose_action()` methods
4. Register in `agents/__init__.py` (import + add to `AVAILABLE_AGENTS` dict)
5. Run: `uv run main.py --agent=my_agent --game=ls20`

### Agent Methods

- **`is_done()`**: Check `latest_frame.state is GameState.WIN` to detect completion
- **`choose_action()`**: Return a `GameAction` (RESET, ACTION1-ACTION6). ACTION6 takes x,y coordinate data.

### Actions
- ACTION1 through ACTION5: Simple actions (game-specific)
- ACTION6: Complex action requiring `data={"x": ..., "y": ...}` coordinates
- ACTION7: Undo action

## Local vs Online Mode

### Local (recommended for development)
```python
from arc_agi import Arcade, OperationMode
arc = Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("ls20", render_mode="terminal")
```
- ~2,000 FPS, unlimited instances, no auth needed, no rate limits
- No online scorecards or replay links

### Online
```python
from arc_agi import Arcade, OperationMode
arc = Arcade(operation_mode=OperationMode.ONLINE)
env = arc.make("ls20", render_mode="terminal")
```
- Scorecards, replays, leaderboard
- Requires API key, 600 RPM rate limit

## Game Schema

- Turn-based environments with 2D grids
- Grid max size: 64x64, cell values: 0-15
- Origin (0,0) at upper-left, (x,y) notation
- Game IDs follow pattern: `<game_name>-<version>` (e.g., `ls20-v1`)

## Game States

- `NOT_FINISHED` - active, awaiting action
- `WIN` - objective completed
- `GAME_OVER` - terminated (max actions exceeded or other condition)

## Scoring: Relative Human Action Efficiency (RHAE)

Per-level score: `(human_baseline_actions / ai_actions) ^ 2`
- Capped at 1.0x per level
- Per-game score: weighted average (level index as weight)
- Total score: average across all games (0-100%)
- Human baseline: 2nd best human performer per game

## Full Play Test API Workflow

1. `GET /api/games` - list available games
2. `POST /api/scorecard/open` - create scorecard (returns `card_id`)
3. `POST /api/cmd/RESET` with `game_id` + `card_id` - start game (returns `guid`)
4. Send actions (ACTION1-ACTION7) - each returns updated state + score
5. `POST /api/scorecard/close` - finalize and get results

## Recordings

- Online: viewable at `https://arcprize.org/scorecards/<scorecard_id>`
- Swarm: saved locally in `recordings/` as JSONL: `{game_id}.{agent_type}.{max_actions}.{guid}.recording.jsonl`
- Local-only mode: no recordings

## Creating Custom Games

Directory structure:
```
ARC-AGI/
└── environment_files/
    └── {gameID}/
        └── v1/
            ├── {gameID}.py
            └── metadata.json
```

Test custom games:
```python
import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade(environments_dir="./my_games")
env = arc.make("ab12-v1", seed=0, render_mode="terminal")
env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
```

Game class must extend `ARCBaseGame` and implement: `__init__`, `on_set_level(level)`, `step()`, `_check_win()`.

## Rate Limits

- 600 RPM during research preview (free, best-effort, no SLA)
- Exceeding returns 429 with exponential backoff
- Contact team@arcprize.org with subject "Increase Rate Limits" for higher throughput

## Useful Links

- Games browser: arcprize.org/tasks
- API keys: arcprize.org/platform
- Docs: docs.arcprize.org
- Docs index: docs.arcprize.org/llms.txt
