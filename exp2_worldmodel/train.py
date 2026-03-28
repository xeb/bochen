"""Train the world model (grid encoder + transition predictor) using JAX/Optax.

Replaces exp2_worldmodel/train.py. Key differences:
- On-the-fly one-hot encoding inside jit (avoids 75GB memory blowup)
- Explicit jax.lax.stop_gradient on target encoder output
- Optax Adam, jax.value_and_grad instead of loss.backward()
- Orbax checkpointing for model params
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exp2_worldmodel.models import GridEncoder, TransitionPredictor
from exp2_worldmodel.config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, VALIDATION_SPLIT
)


def load_data(data_paths: list[str]) -> tuple[dict, dict]:
    """Load raw transition data (int8 grids, int actions). No one-hot encoding.

    Returns train_data, val_data dicts with numpy arrays.
    Memory: ~1.2 GB for 150k transitions (vs ~75 GB with pre-encoded one-hot).
    """
    all_before = []
    all_actions = []
    all_after = []

    for path in data_paths:
        data = np.load(path)
        all_before.append(data["grids_before"])
        all_actions.append(data["actions"])
        all_after.append(data["grids_after"])

    grids_before = np.concatenate(all_before)
    actions = np.concatenate(all_actions).astype(np.int32)
    grids_after = np.concatenate(all_after)

    n = len(grids_before)
    split = int(n * (1 - VALIDATION_SPLIT))

    train_data = {
        "before": grids_before[:split],
        "actions": actions[:split],
        "after": grids_after[:split],
    }
    val_data = {
        "before": grids_before[split:],
        "actions": actions[split:],
        "after": grids_after[split:],
    }
    return train_data, val_data


def _one_hot_batch(grids_int, num_colors=16, pad_to=64):
    """One-hot encode a batch of int grids inside JAX.

    grids_int: (B, H, W) int array
    Returns: (B, num_colors, pad_to, pad_to) float32 in NCHW format
    (matching existing perception.one_hot_grid API)
    """
    B = grids_int.shape[0]
    h, w = grids_int.shape[1], grids_int.shape[2]
    # One-hot to (B, H, W, C) then pad spatial dims then transpose to NCHW
    oh = jax.nn.one_hot(grids_int, num_colors)  # (B, H, W, C)
    # Pad to (B, pad_to, pad_to, C)
    pad_h = pad_to - h
    pad_w = pad_to - w
    if pad_h > 0 or pad_w > 0:
        oh = jnp.pad(oh, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    # Transpose to NCHW to match model input convention
    return jnp.transpose(oh, (0, 3, 1, 2))


def train_world_model(data_paths: list[str], save_dir: str = None) -> dict:
    """Train encoder + transition predictor. Returns metrics dict."""
    save_dir = save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "exp2_worldmodel", "checkpoints"
    )
    os.makedirs(save_dir, exist_ok=True)

    print("  [exp2 train/jax] Loading data...")
    train_data, val_data = load_data(data_paths)
    n_train = len(train_data["before"])
    n_val = len(val_data["before"])
    print(f"  [exp2 train/jax] {n_train} train, {n_val} val transitions")

    # Initialize models
    encoder = GridEncoder()
    predictor = TransitionPredictor(state_dim=128, action_dim=8)

    rng = jax.random.key(42)
    rng_enc, rng_pred = jax.random.split(rng)

    # Dummy inputs for init
    dummy_grid = jnp.zeros((1, 16, 64, 64))
    dummy_state = jnp.zeros((1, 128))
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)

    params = {
        "encoder": encoder.init(rng_enc, dummy_grid),
        "predictor": predictor.init(rng_pred, dummy_state, dummy_action),
    }

    tx = optax.adam(learning_rate=TRAIN_LR)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_before, batch_after, batch_actions):
        grids_before = _one_hot_batch(batch_before)
        grids_after = _one_hot_batch(batch_after)

        def loss_fn(params):
            state_enc = encoder.apply(params["encoder"], grids_before)
            # stop_gradient: target encoder output must not receive gradients
            # (matches .detach() in the PyTorch version)
            target_enc = jax.lax.stop_gradient(
                encoder.apply(params["encoder"], grids_after)
            )
            preds = predictor.apply(params["predictor"], state_enc, batch_actions)
            return jnp.mean((preds["next_state"] - target_enc) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def val_step(params, batch_before, batch_after, batch_actions):
        grids_before = _one_hot_batch(batch_before)
        grids_after = _one_hot_batch(batch_after)

        state_enc = encoder.apply(params["encoder"], grids_before)
        target_enc = encoder.apply(params["encoder"], grids_after)
        preds = predictor.apply(params["predictor"], state_enc, batch_actions)

        val_loss = jnp.mean((preds["next_state"] - target_enc) ** 2)
        cos_sim = jnp.mean(
            jnp.sum(preds["next_state"] * target_enc, axis=-1) /
            (jnp.linalg.norm(preds["next_state"], axis=-1) *
             jnp.linalg.norm(target_enc, axis=-1) + 1e-8)
        )
        return val_loss, cos_sim

    best_val_acc = 0.0
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(TRAIN_EPOCHS):
        # Shuffle
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = perm[i:i + TRAIN_BATCH_SIZE]
            b_before = jnp.array(train_data["before"][idx])
            b_after = jnp.array(train_data["after"][idx])
            b_actions = jnp.array(train_data["actions"][idx])

            params, opt_state, loss = train_step(
                params, opt_state, b_before, b_after, b_actions
            )
            epoch_loss += float(loss)
            n_batches += 1

        # Validation
        val_loss, cos_sim = val_step(
            params,
            jnp.array(val_data["before"]),
            jnp.array(val_data["after"]),
            jnp.array(val_data["actions"]),
        )
        val_loss = float(val_loss)
        cos_sim = float(cos_sim)

        avg_train_loss = epoch_loss / max(1, n_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [exp2 train/jax] Epoch {epoch+1}/{TRAIN_EPOCHS}: "
                  f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_cosine_sim={cos_sim:.4f}")

        if cos_sim > best_val_acc:
            best_val_acc = cos_sim
            save_path = os.path.join(save_dir, "world_model_best_jax")
            checkpointer.save(
                save_path,
                {
                    "encoder": params["encoder"],
                    "predictor": params["predictor"],
                    "epoch": epoch,
                    "val_cosine_sim": cos_sim,
                },
                force=True,
            )

    print(f"  [exp2 train/jax] Best val cosine similarity: {best_val_acc:.4f}")
    return {
        "best_val_cosine_sim": best_val_acc,
        "final_train_loss": avg_train_loss,
        "params": params,
    }
