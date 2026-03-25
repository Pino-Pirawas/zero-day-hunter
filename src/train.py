import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import copy


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


def val_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                epochs: int = 100,
                patience: int = 10) -> Tuple[nn.Module, Dict]:

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status'}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            status = "⭐ best"
        else:
            patience_counter += 1
            status = f"no improve ({patience_counter}/{patience})"

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {status}")

        if patience_counter >= patience:
            print(f"\n⚡ Early stopping triggered at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"\n✅ Training complete — best val loss: {best_val_loss:.6f}")
    return model, history
