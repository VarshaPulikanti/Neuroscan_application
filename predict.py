"""Run inference on a single brain MRI image."""

import argparse
from pathlib import Path

from src.inference import InferenceEngine


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

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    engine = InferenceEngine(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    engine.load()

    image = InferenceEngine.bytes_to_image(image_path.read_bytes())

    if args.gradcam:
        result, overlay = engine.predict_with_gradcam(image)
        output_path = Path(
            args.output
            or Path(engine.config["output_dir"]) / f"gradcam_{image_path.stem}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(output_path)
        print(f"\nGrad-CAM saved to: {output_path}")
    else:
        result = engine.predict(image)

    print(f"Prediction: {result.prediction_label}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print("\nAll classes:")
    for label, prob in result.probabilities.items():
        print(f"  {label}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
