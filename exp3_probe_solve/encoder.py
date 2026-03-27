"""State encoder CNN for causal memory embeddings."""

import torch
import torch.nn as nn
import numpy as np


class StateEncoder(nn.Module):
    """Encode grid with 16 colors into fixed-dim embedding."""

    def __init__(self, embed_dim=128, num_colors=16, pad_to=64):
        super().__init__()
        self.pad_to = pad_to
        self.num_colors = num_colors
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)

    def encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """Encode a single numpy grid to embedding vector."""
        h, w = grid.shape
        onehot = np.zeros((self.num_colors, self.pad_to, self.pad_to), dtype=np.float32)
        for c in range(self.num_colors):
            onehot[c, :h, :w] = (grid == c).astype(np.float32)
        x = torch.from_numpy(onehot).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            return self(x).squeeze(0)
