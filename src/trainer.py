"""Training loop with early stopping."""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import evaluate_model, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        learning_rate: float = 0.0003,
        weight_decay: float = 1e-4,
        patience: int = 5,
        checkpoint_path: Path | None = None,
        config: dict | None = None,
        class_names: list[str] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.config = config or {}
        self.class_names = class_names or []

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, 100.0 * correct / total

    def fit(self, epochs: int) -> dict:
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, _, _ = evaluate_model(
                self.model, self.val_loader, self.device, self.criterion
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_acc)
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                if self.checkpoint_path:
                    save_checkpoint(
                        self.checkpoint_path,
                        self.model,
                        self.optimizer,
                        epoch,
                        val_acc,
                        self.config,
                        self.class_names,
                    )
                    print(f"  Saved best model (val acc: {val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        return self.history
