"""Causal memory with numpy storage and JAX-accelerated retrieval.

Replaces exp3_probe_solve/memory.py. Key differences:
- Embeddings stored as numpy (mutable, no copy-on-write overhead)
- JAX used only for the retrieval matmul (where it actually helps)
- Persistence stays as pickle (mixed Python dicts + numpy, not a pure pytree)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle

from exp3_probe_solve.encoder import encode_grid


class CausalMemory:
    """Causal memory for cross-game transfer with JAX-accelerated retrieval."""

    def __init__(self, encoder_params: dict, max_entries: int = 10000,
                 embed_dim: int = 128):
        self.encoder_params = encoder_params
        self.embed_dim = embed_dim
        self.max_entries = max_entries
        self.entries = []
        self.embeddings = np.zeros((max_entries, embed_dim), dtype=np.float32)
        self.count = 0

    def store(self, grid: np.ndarray, action: str, effect: dict, game_id: str):
        """Store an observation triple."""
        emb = encode_grid(self.encoder_params, grid)
        self.entries.append({
            "action": action,
            "effect": effect,
            "game_id": game_id,
            "confirmed_count": 1,
        })
        self.embeddings[self.count] = emb
        self.count += 1

    def retrieve(self, grid: np.ndarray, k: int = 10) -> list[tuple[dict, float]]:
        """Find K most similar memories by state embedding cosine similarity."""
        if self.count == 0:
            return []
        query = encode_grid(self.encoder_params, grid)  # (embed_dim,)
        actual_k = min(k, self.count)

        # Move to JAX only for the matmul
        q = jnp.array(query)
        emb = jnp.array(self.embeddings[:self.count])
        sims = jnp.dot(emb, q)  # (count,)
        top_vals, top_idx = jax.lax.top_k(sims, actual_k)

        indices = top_idx.tolist()
        values = top_vals.tolist()
        return [(self.entries[i], values[j]) for j, i in enumerate(indices)]

    def retrieve_for_action(self, grid: np.ndarray, action: str,
                            k: int = 5) -> list[tuple[dict, float]]:
        """Retrieve memories for a specific action type."""
        all_results = self.retrieve(grid, k=k * 3)
        filtered = [(e, s) for e, s in all_results if e["action"] == action]
        return filtered[:k]

    def get_known_effects(self) -> dict:
        """Summarize what we know about each action across all memories."""
        effects_by_action = {}
        for entry in self.entries:
            action = entry["action"]
            effects_by_action.setdefault(action, [])
            effects_by_action[action].append(entry["effect"])
        return effects_by_action

    def save(self, path: str):
        """Serialize memory to disk. Uses pickle (not Orbax) because entries
        contain arbitrary Python dicts with strings and nested structures."""
        data = {
            "entries": self.entries,
            "embeddings": self.embeddings[:self.count],
            "count": self.count,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load memory from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.entries = data["entries"]
        self.count = data["count"]
        self.embeddings[:self.count] = data["embeddings"]

    @property
    def size(self) -> int:
        return self.count
