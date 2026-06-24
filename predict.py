"""Run inference on a single brain MRI image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD, load_config
from src.gradcam import GradCAM, get_gradcam_target_layer, overlay_heatmap
from src.utils import build_model, get_device, load_checkpoint

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict brain MRI tumor class")
    parser.add_argument("image", type=str, help="Path to MRI image")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Save Grad-CAM heatmap overlay",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for Grad-CAM image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = Path(
        args.checkpoint
        or Path(config["output_dir"]) / "checkpoints" / config["checkpoint_name"]
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\nTrain first with: python train.py"
        )

    checkpoint = load_checkpoint(
        checkpoint_path,
        build_model(config["model"], config["num_classes"]),
        device,
    )
    model_name = checkpoint["config"].get("model", config["model"])
    class_names = checkpoint.get("class_names", config["class_names"])
    model = build_model(model_name, config["num_classes"])
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    image_size = config.get("image_size", 224)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_idx = probabilities.argmax().item()
    predicted = class_names[top_idx]
    display_name = CLASS_LABELS.get(predicted, predicted)

    print(f"Prediction: {display_name}")
    print(f"Confidence: {probabilities[top_idx].item() * 100:.1f}%")
    print("\nAll classes:")
    for idx, name in enumerate(class_names):
        label = CLASS_LABELS.get(name, name)
        print(f"  {label}: {probabilities[idx].item() * 100:.1f}%")

    if args.gradcam:
        target_layer = get_gradcam_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)
        try:
            model.eval()
            tensor_grad = transform(image).unsqueeze(0).to(device)
            tensor_grad.requires_grad_(True)
            heatmap = gradcam.generate(tensor_grad, class_idx=top_idx)
            overlay = overlay_heatmap(image, heatmap)

            output_path = Path(
                args.output or Path(config["output_dir"]) / f"gradcam_{image_path.stem}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(output_path)
            print(f"\nGrad-CAM saved to: {output_path}")
        finally:
            gradcam.close()


if __name__ == "__main__":
    main()
