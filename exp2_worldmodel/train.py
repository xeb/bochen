"""Train the world model (grid encoder + transition predictor) on collected data."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.perception import one_hot_grid
from exp2_worldmodel.models import GridEncoder, TransitionPredictor
from exp2_worldmodel.config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, VALIDATION_SPLIT
)


def load_data(data_paths: list[str], device: str = "cuda"):
    """Load and preprocess collected transition data."""
    all_before = []
    all_actions = []
    all_after = []

    for path in data_paths:
        data = np.load(path)
        all_before.append(data["grids_before"])
        all_actions.append(data["actions"])
        all_after.append(data["grids_after"])

    grids_before = np.concatenate(all_before)
    actions = np.concatenate(all_actions)
    grids_after = np.concatenate(all_after)

    # One-hot encode
    n = len(grids_before)
    X_before = np.stack([one_hot_grid(g) for g in grids_before])
    X_after = np.stack([one_hot_grid(g) for g in grids_after])

    # Split
    split = int(n * (1 - VALIDATION_SPLIT))
    train_data = {
        "before": torch.from_numpy(X_before[:split]).to(device),
        "actions": torch.from_numpy(actions[:split].astype(np.int64)).to(device),
        "after": torch.from_numpy(X_after[:split]).to(device),
    }
    val_data = {
        "before": torch.from_numpy(X_before[split:]).to(device),
        "actions": torch.from_numpy(actions[split:].astype(np.int64)).to(device),
        "after": torch.from_numpy(X_after[split:]).to(device),
    }
    return train_data, val_data


def train_world_model(data_paths: list[str], save_dir: str = None,
                      device: str = "cuda") -> dict:
    """Train encoder + transition predictor. Returns metrics dict."""
    save_dir = save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
    os.makedirs(save_dir, exist_ok=True)

    print("  [exp2 train] Loading data...")
    train_data, val_data = load_data(data_paths, device)
    n_train = len(train_data["before"])
    n_val = len(val_data["before"])
    print(f"  [exp2 train] {n_train} train, {n_val} val transitions")

    encoder = GridEncoder().to(device)
    predictor = TransitionPredictor(state_dim=128, action_dim=8).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=TRAIN_LR
    )

    best_val_acc = 0.0

    for epoch in range(TRAIN_EPOCHS):
        encoder.train()
        predictor.train()

        # Shuffle
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, TRAIN_BATCH_SIZE):
            idx = perm[i:i + TRAIN_BATCH_SIZE]
            x_before = train_data["before"][idx]
            acts = train_data["actions"][idx]
            x_after = train_data["after"][idx]

            state_enc = encoder(x_before)
            target_enc = encoder(x_after)
            preds = predictor(state_enc, acts)

            # Loss: predict next state embedding
            loss = nn.functional.mse_loss(preds["next_state"], target_enc.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            val_enc = encoder(val_data["before"])
            val_target = encoder(val_data["after"])
            val_preds = predictor(val_enc, val_data["actions"])
            val_loss = nn.functional.mse_loss(val_preds["next_state"], val_target).item()

            # Accuracy proxy: cosine similarity between predicted and actual
            cos_sim = nn.functional.cosine_similarity(
                val_preds["next_state"], val_target, dim=-1
            ).mean().item()

        avg_train_loss = epoch_loss / max(1, n_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [exp2 train] Epoch {epoch+1}/{TRAIN_EPOCHS}: "
                  f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_cosine_sim={cos_sim:.4f}")

        if cos_sim > best_val_acc:
            best_val_acc = cos_sim
            torch.save({
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "epoch": epoch,
                "val_cosine_sim": cos_sim,
            }, os.path.join(save_dir, "world_model_best.pt"))

    print(f"  [exp2 train] Best val cosine similarity: {best_val_acc:.4f}")
    return {
        "best_val_cosine_sim": best_val_acc,
        "final_train_loss": avg_train_loss,
        "encoder_path": os.path.join(save_dir, "world_model_best.pt"),
    }
