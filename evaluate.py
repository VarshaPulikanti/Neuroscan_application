"""Evaluate a trained brain MRI classifier on the test set."""

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data import get_dataloaders
from src.utils import (
    build_model,
    evaluate_model,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NeuroScan classifier")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    checkpoint_path = Path(
        args.checkpoint
        or Path(config["output_dir"]) / "checkpoints" / config["checkpoint_name"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python train.py"
        )

    _, _, test_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config.get("image_size", 224),
        train_split=config.get("train_dir", "Training"),
        test_split=config.get("test_dir", "Testing"),
        val_split=config.get("val_split", 0.15),
        num_workers=config.get("num_workers", 0),
    )

    checkpoint_meta = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint_meta["config"].get("model", config["model"])
    model = build_model(model_name, config["num_classes"])
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    class_names = checkpoint.get("class_names", config["class_names"])

    loss, accuracy, y_true, y_pred = evaluate_model(
        model, test_loader, device, criterion=None
    )

    output_dir = Path(config["output_dir"])
    report = plot_confusion_matrix(
        y_true,
        y_pred,
        output_dir / "confusion_matrix.png",
        class_names=class_names,
    )

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion matrix saved to: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
