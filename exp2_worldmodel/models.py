"""JAX/Flax neural network models for the world model experiment.

Replaces exp2_worldmodel/models.py. Omits ObjectSegmenter (unused at runtime,
BatchNorm makes the Flax port disproportionately complex).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


def _adaptive_avg_pool(x, output_size):
    """Replicate PyTorch's AdaptiveAvgPool2d for channel-last (B, H, W, C) input.

    Only works when H and W are evenly divisible by the target size.
    Our grids are always 64x64, pooling to 16 or 4 — both divide evenly.
    """
    B, H, W, C = x.shape
    oh, ow = output_size
    wh, ww = H // oh, W // ow
    x = x.reshape(B, oh, wh, ow, ww, C)
    return x.mean(axis=(2, 4))


class GridEncoder(nn.Module):
    """Encode full grid into a fixed-size state vector.

    Input:  (B, 16, 64, 64) one-hot grid in NCHW format (matching existing API)
    Output: (B, out_dim) state embedding
    """
    num_colors: int = 16
    out_dim: int = 128

    @nn.compact
    def __call__(self, x):
        # Transpose NCHW -> NHWC for Flax convolutions
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = _adaptive_avg_pool(x, (16, 16))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = _adaptive_avg_pool(x, (4, 4))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.out_dim)(x)
        return x


class TransitionPredictor(nn.Module):
    """Predict object-level transitions given state features and action.

    Input:  state_features (B, state_dim), action_idx (B,) int
    Output: dict with delta_xy, new_color, delete_prob, next_state
    """
    state_dim: int = 128
    action_dim: int = 8
    hidden: int = 256
    num_layers: int = 3

    @nn.compact
    def __call__(self, state_features, action_idx):
        act_emb = nn.Embed(num_embeddings=self.action_dim, features=32)(action_idx)
        x = jnp.concatenate([state_features, act_emb], axis=-1)

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden)(x)
            x = nn.relu(x)

        return {
            "delta_xy": nn.Dense(features=2, name="delta_head")(x),
            "new_color": nn.Dense(features=16, name="color_head")(x),
            "delete_prob": jax.nn.sigmoid(nn.Dense(features=1, name="delete_head")(x)),
            "next_state": nn.Dense(features=128, name="state_head")(x),
        }
