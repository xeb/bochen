"""JAX Experiment 3: Two-Phase Probe-Solve Agent with Causal Memory."""

import sys
import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.env_wrapper import ArcEnv
from exp3_probe_solve.encoder import StateEncoder
from exp3_probe_solve.memory import CausalMemory
from exp3_probe_solve.probe import run_probe_battery
from exp3_probe_solve.solver import solve
from exp3_probe_solve.config import MEMORY_EMBED_DIM


class ProbeSolveAgent:
    """Two-phase agent: systematic probing then memory-guided solving (JAX)."""

    def __init__(self, memory_path: str = None):
        self.se = StateEncoder(embed_dim=MEMORY_EMBED_DIM)
        rng = jax.random.key(42)
        dummy = jnp.zeros((1, 16, 64, 64))
        self.encoder_params = self.se.init(rng, dummy)

        self.memory = CausalMemory(
            encoder_params=self.encoder_params,
            max_entries=10000,
            embed_dim=MEMORY_EMBED_DIM,
        )

        if memory_path and os.path.exists(memory_path):
            self.memory.load(memory_path)
            print(f"  [exp3/jax] Loaded memory with {self.memory.size} entries")

    def run_episode(self, game_id: str) -> dict:
        """Run one full probe-then-solve episode."""
        env = ArcEnv(game_id, offline=False, render=False)

        # Phase 1: Probe
        probe_log = run_probe_battery(env, self.memory, game_id)
        probe_actions = len([e for e in probe_log
                            if "REPEAT" not in e.get("action", "")
                            and "UNDO" not in e.get("action", "")])

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
        """Save encoder params via Orbax."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ocp.StandardCheckpointer().save(path, self.encoder_params, force=True)
