"""PyTorch training loop — builds model, trains, returns TrainingResult."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple

from src.domain.automl.architecture import build_model, count_parameters
from src.domain.entities.result import TrainingResult


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    config: Dict[str, Any],
    input_dim: int,
    output_dim: int,
    device: str = "cpu",
) -> TrainingResult:
    """
    Train a model defined by `config` and return a TrainingResult.
    """
    device = torch.device(device)
    model = build_model(input_dim, output_dim, config).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_map = {
        "adam":    optim.Adam,
        "adamw":   optim.AdamW,
        "sgd":     optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    optimizer_cls = opt_map.get(config.get("optimizer", "adam"), optim.Adam)
    optimizer = optimizer_cls(model.parameters(), lr=config.get("lr", 1e-3))

    # ── Loss ──────────────────────────────────────────────────────────────────
    is_classification = output_dim > 1
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_ds = TensorDataset(X_train.to(device), y_train.to(device))
    train_dl = DataLoader(train_ds, batch_size=config.get("batch_size", 64), shuffle=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs = config.get("epochs", 10)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    training_time = time.time() - t0

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        # Validation
        val_out  = model(X_val.to(device))
        val_loss = criterion(val_out, y_val.to(device)).item()

        if is_classification:
            val_preds  = val_out.argmax(dim=1)
            val_acc    = (val_preds == y_val.to(device)).float().mean().item()
        else:
            val_acc = max(0.0, 1.0 - val_loss)  # pseudo-accuracy for regression

        # Train (last epoch)
        train_out  = model(X_train.to(device))
        train_loss = criterion(train_out, y_train.to(device)).item()
        if is_classification:
            train_acc = (train_out.argmax(1) == y_train.to(device)).float().mean().item()
        else:
            train_acc = max(0.0, 1.0 - train_loss)

    return TrainingResult(
        val_accuracy     = round(val_acc, 4),
        val_loss         = round(val_loss, 4),
        train_accuracy   = round(train_acc, 4),
        train_loss       = round(train_loss, 4),
        epochs_run       = epochs,
        training_time_sec= round(training_time, 2),
        model_params     = count_parameters(model),
    )