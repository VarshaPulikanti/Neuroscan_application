"""Train a brain MRI tumor classifier."""

import argparse
from pathlib import Path

from src.config import load_config
from src.data import get_dataloaders
from src.trainer import Trainer
from src.utils import build_model, get_device, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuroScan brain MRI classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--model",
        type=str,
        choices=["efficientnet_b0", "resnet18"],
        default=None,
        help="Override model architecture",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model:
        config["model"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {config['model']}")

    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / config["checkpoint_name"]

    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config.get("image_size", 224),
        train_split=config.get("train_dir", "Training"),
        test_split=config.get("test_dir", "Testing"),
        val_split=config.get("val_split", 0.15),
        num_workers=config.get("num_workers", 0),
    )

    print(f"Classes: {class_names}")

    model = build_model(config["model"], config["num_classes"])
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
        patience=config.get("patience", 5),
        checkpoint_path=checkpoint_path,
        config=config,
        class_names=class_names,
    )

    history = trainer.fit(config["epochs"])

    plot_training_history(history, output_dir / "training_history.png")
    print(f"\nTraining complete. Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training plot saved to: {output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()
