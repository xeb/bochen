"""JAX/Flax state encoder CNN for causal memory embeddings.

Replaces exp3_probe_solve/encoder.py.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


def _adaptive_avg_pool(x, output_size):
    """Channel-last (B, H, W, C) adaptive average pooling."""
    B, H, W, C = x.shape
    oh, ow = output_size
    wh, ww = H // oh, W // ow
    x = x.reshape(B, oh, wh, ow, ww, C)
    return x.mean(axis=(2, 4))


class StateEncoder(nn.Module):
    """Encode grid with 16 colors into fixed-dim L2-normalized embedding.

    Input:  (B, 16, 64, 64) one-hot grid in NCHW format
    Output: (B, embed_dim) unit-normalized embedding
    """
    embed_dim: int = 128
    num_colors: int = 16

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = _adaptive_avg_pool(x, (16, 16))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = _adaptive_avg_pool(x, (4, 4))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.embed_dim)(x)
        # L2 normalize (matches PyTorch nn.functional.normalize(x, dim=-1))
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x


def encode_grid(params: dict, grid: np.ndarray,
                num_colors: int = 16, pad_to: int = 64) -> np.ndarray:
    """Encode a single numpy grid to embedding vector. Returns numpy (embed_dim,).

    This is the JAX equivalent of StateEncoder.encode_grid() — a standalone
    function since Flax modules are stateless.
    """
    h, w = grid.shape
    onehot = np.zeros((num_colors, pad_to, pad_to), dtype=np.float32)
    for c in range(num_colors):
        onehot[c, :h, :w] = (grid == c).astype(np.float32)

    x = jnp.array(onehot)[None, ...]  # (1, C, H, W)
    emb = _encode_jit(params, x)
    return np.asarray(emb[0])


@jax.jit
def _encode_jit(params, x):
    """JIT-compiled forward pass for StateEncoder."""
    return StateEncoder().apply(params, x)
