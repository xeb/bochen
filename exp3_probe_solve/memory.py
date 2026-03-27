"""Causal memory: stores (state, action, effect) triples with GPU-accelerated retrieval."""

import torch
import numpy as np
import pickle
from exp3_probe_solve.encoder import StateEncoder


class CausalMemory:
    """GPU-accelerated causal memory for cross-game transfer."""

    def __init__(self, encoder: StateEncoder, device: str = "cuda"):
        self.encoder = encoder
        self.device = device
        self.entries = []          # list of dicts
        self.embeddings = None     # (N, embed_dim) tensor on GPU

    def store(self, grid: np.ndarray, action: str, effect: dict, game_id: str):
        """Store an observation triple."""
        emb = self.encoder.encode_grid(grid)
        self.entries.append({
            "action": action,
            "effect": effect,
            "game_id": game_id,
            "confirmed_count": 1,
        })
        if self.embeddings is None:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)])

    def retrieve(self, grid: np.ndarray, k: int = 10) -> list[tuple[dict, float]]:
        """Find K most similar memories by state embedding cosine similarity."""
        if self.embeddings is None or len(self.entries) == 0:
            return []
        query = self.encoder.encode_grid(grid).unsqueeze(0)
        sims = torch.mm(query, self.embeddings.T).squeeze(0)
        actual_k = min(k, len(self.entries))
        topk = torch.topk(sims, actual_k)
        return [
            (self.entries[i], sims[i].item())
            for i in topk.indices.tolist()
        ]

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
        """Serialize memory to disk."""
        data = {
            "entries": self.entries,
            "embeddings": self.embeddings.cpu().numpy() if self.embeddings is not None else None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load memory from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.entries = data["entries"]
        if data["embeddings"] is not None:
            self.embeddings = torch.from_numpy(data["embeddings"]).to(self.device)

    @property
    def size(self) -> int:
        return len(self.entries)
