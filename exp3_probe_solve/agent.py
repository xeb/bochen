"""Experiment 3: Two-Phase Probe-Solve Agent with Causal Memory."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.env_wrapper import ArcEnv
from exp3_probe_solve.encoder import StateEncoder
from exp3_probe_solve.memory import CausalMemory
from exp3_probe_solve.probe import run_probe_battery
from exp3_probe_solve.solver import solve
from exp3_probe_solve.config import MEMORY_EMBED_DIM


class ProbeSolveAgent:
    """Two-phase agent: systematic probing then memory-guided solving."""

    def __init__(self, device: str = "cuda", memory_path: str = None):
        self.device = device
        self.encoder = StateEncoder(embed_dim=MEMORY_EMBED_DIM).to(device)
        self.memory = CausalMemory(self.encoder, device)

        if memory_path and os.path.exists(memory_path):
            self.memory.load(memory_path)
            print(f"  [exp3] Loaded memory with {self.memory.size} entries")

    def run_episode(self, game_id: str) -> dict:
        """Run one full probe-then-solve episode."""
        env = ArcEnv(game_id, offline=True, render=False)

        # Phase 1: Probe
        probe_log = run_probe_battery(env, self.memory, game_id)
        probe_actions = len([e for e in probe_log
                            if "REPEAT" not in e.get("action", "")
                            and "UNDO" not in e.get("action", "")])

        # Analyze probe results
        effective = sum(1 for e in probe_log
                       if e.get("effect", {}).get("cells_changed", 0) > 0)

        # Phase 2: Solve
        won, solve_actions, level = solve(env, self.memory, probe_log, game_id)

        total_actions = probe_actions + solve_actions
        return {
            "won": won,
            "level": level,
            "actions": total_actions,
            "probe_actions": probe_actions,
            "solve_actions": solve_actions,
            "game_id": game_id,
            "memory_size": self.memory.size,
            "effective_probes": effective,
            "total_probes": len(probe_log),
        }

    def save_memory(self, path: str):
        """Save causal memory to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.memory.save(path)

    def save_encoder(self, path: str):
        """Save encoder weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
