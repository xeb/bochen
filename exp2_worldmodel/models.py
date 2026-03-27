"""Neural network models for the world model experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectSegmenter(nn.Module):
    """CNN that predicts per-cell object membership from one-hot grid."""

    def __init__(self, num_colors=16, channels=None):
        super().__init__()
        channels = channels or [16, 32, 64]
        layers = []
        in_c = num_colors
        for out_c in channels:
            layers.extend([
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ])
            in_c = out_c
        # Output: per-cell embedding for clustering into objects
        layers.append(nn.Conv2d(in_c, 32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Input: (B, 16, H, W), Output: (B, 32, H, W) per-cell embeddings."""
        return self.net(x)


class TransitionPredictor(nn.Module):
    """Predict object-level transitions given state features and action."""

    def __init__(self, state_dim=128, action_dim=8, hidden=256, num_layers=3):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, 32)
        layers = []
        in_dim = state_dim + 32
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        # Heads
        self.delta_head = nn.Linear(hidden, 2)       # dx, dy movement
        self.color_head = nn.Linear(hidden, 16)       # new color (softmax)
        self.delete_head = nn.Linear(hidden, 1)       # P(deleted)
        self.state_head = nn.Linear(hidden, 128)      # predicted next state embedding

    def forward(self, state_features, action_idx):
        """
        state_features: (B, state_dim)
        action_idx: (B,) long tensor
        """
        act_emb = self.action_embed(action_idx)
        x = torch.cat([state_features, act_emb], dim=-1)
        h = self.backbone(x)
        return {
            "delta_xy": self.delta_head(h),
            "new_color": self.color_head(h),
            "delete_prob": torch.sigmoid(self.delete_head(h)),
            "next_state": self.state_head(h),
        }


class GridEncoder(nn.Module):
    """Encode full grid into a fixed-size state vector."""

    def __init__(self, num_colors=16, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
        )

    def forward(self, x):
        return self.net(x)
