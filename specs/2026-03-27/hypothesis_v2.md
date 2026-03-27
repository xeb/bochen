# Revised Hypothesis — After 178 Failed Episodes

## What We Learned

### The games are spatial puzzles
ls20 is a Sokoban-like sliding puzzle:
- 64x64 grid with corridors (color 3) and background (color 4)
- Colored blocks (9, 12) that slide when actions are taken
- ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT
- 7 levels to beat, each presumably harder
- No hard action limit — the game waits for you to solve it

### Why every agent failed (0 wins in 178 episodes)
1. **exp1 (LLM Scientist)**: Correctly identified "ACTION1 moves objects" at 0.99 confidence. Never figured out WHERE they need to go. The LLM doesn't see the spatial structure — it gets color histograms and object counts, not the spatial layout that reveals the puzzle logic.

2. **exp2 (World Model)**: Trained on 6000 random transitions. Learned to predict state changes. But prediction accuracy doesn't help if you don't know the GOAL. The info-gain search explored novel states but never aimed at winning.

3. **exp3 (Probe-Solve)**: Found 5 effective actions per game. Solver picked the action with highest score_delta, but score never changed (it's binary: 0 until you complete a level). So the solver was effectively random.

### The real blocker: API rate limit kills search
- BFS over 4 actions at depth 12 = 16M states to explore
- At 300 RPM = 37 days to search exhaustively
- Even the harness only managed ~2 API calls/sec
- The GPU sat nearly idle while agents waited for API responses

## The New Hypothesis

**The winning strategy is: use a few API calls to learn game mechanics, then do ALL planning on the GPU using a world model, and only call the API to execute the final solution.**

### Concrete plan: "Imagine Then Act"

```
Phase 1: LEARN (50-100 API calls per game)
  - Systematic probes to discover action effects
  - Record (state, action, next_state) triples
  - Identify: movable objects, walls, targets, action directions

Phase 2: BUILD MODEL (GPU, 0 API calls)
  - Train a deterministic transition model from probe data
  - Since games are deterministic, even 50 transitions can be enough
  - Validate model by predicting held-out transitions

Phase 3: SEARCH IN IMAGINATION (GPU, 0 API calls)
  - BFS/IDDFS/A* in the world model's predicted states
  - 4 actions × depth 20 = ~1 trillion states in theory
  - But with state deduplication + pruning = tractable
  - RTX 4090 can evaluate millions of model forward passes per second
  - A* with a learned heuristic (distance to goal state) prunes massively

Phase 4: EXECUTE (few API calls)
  - Take the solution found in imagination
  - Execute it step by step via API
  - If model was wrong at any step, go back to Phase 1 with new data
```

### Why this should work
- **GPU utilization**: The GPU was at 0% during episodes because all time was spent waiting for API. This approach puts 99% of compute on GPU.
- **Deterministic games**: Unlike Atari where randomness makes models useless after a few steps, ARC-AGI-3 games are deterministic. A correct model enables PERFECT planning.
- **Small state space**: The actual reachable state space of a Sokoban puzzle with 2-3 blocks and 4 directions is typically 10K-100K states. BFS on GPU at millions of states/sec solves this in milliseconds.
- **Rate limit irrelevant**: Phase 1 uses ~100 API calls. Phase 4 uses ~20 (the solution length). Total: ~120 API calls per game level. Well under 600 RPM.

### What the world model needs to learn
For Sokoban-like games:
1. Which cells are walls (don't change when you push toward them)
2. Which cells are movable objects (change position on action)
3. Movement rules (objects slide until hitting a wall or another object)
4. Win condition (all objects on target cells, or specific arrangement)

For other game types:
- The same learn-model-search loop, but the model learns different dynamics
- The key is that the model is per-game, trained from probes ON that specific game

### Implementation: what to change

**Keep**: harness.py, db.py, notify.py, shared/ — all working great

**Replace agent logic**:
```python
class ImagineAgent:
    def run_episode(self, game_id):
        # Phase 1: Probe (on API)
        transitions = self.probe(game_id, budget=100)

        # Phase 2: Train model (on GPU)
        model = self.train_model(transitions)  # ~1 second

        # Phase 3: Search in imagination (on GPU)
        solution = self.bfs_in_model(model)  # milliseconds to seconds

        # Phase 4: Execute (on API)
        if solution:
            return self.execute(game_id, solution)
        else:
            return {"won": False, "reason": "no solution found in model"}
```

**World model architecture**:
- Input: 64x64 grid (one-hot 16 colors) + action (one-hot 4-7)
- Output: 64x64 grid (predicted next state)
- Architecture: small U-Net or even just a lookup table for small games
- Training: 50-100 (state, action, next_state) pairs
- Loss: per-cell cross-entropy

**Search algorithm**:
- BFS with state hashing (dedup visited states)
- Win detection: check if predicted state matches any "won" pattern
- Problem: we don't know the win condition from probes alone
- Solution: try BFS to find states that are "maximally different from start" or that minimize a learned value function

### Expected improvement
- API calls per episode: ~120 (was ~80 with no planning)
- GPU utilization: 90%+ during search phase (was 0%)
- Episodes to first win: 1-5 (if model is accurate and search finds solution)
- Time per episode: ~60s probe + ~1s train + ~1s search + ~20s execute = ~82s

### Risks
- Win condition is unknown — we need a way to detect goals from probes
- Model might be inaccurate on unseen states far from training distribution
- Some games may not be deterministic (though docs say they are)
- Some games may require ACTION6 (coordinate clicks) making search space huge
