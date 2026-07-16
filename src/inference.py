"""Shared inference service for brain MRI classification."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
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


@dataclass
class PredictionResult:
    prediction: str
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]
    top_idx: int


class InferenceEngine:
    def __init__(
        self,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        self.config = load_config(config_path)
        self.device = get_device()
        env_checkpoint = os.getenv("CHECKPOINT_PATH")
        self._checkpoint_path = Path(
            checkpoint_path
            or env_checkpoint
            or Path(self.config["output_dir"])
            / "checkpoints"
            / self.config["checkpoint_name"]
        )
        self._model = None
        self._class_names: list[str] = []
        self._model_name = ""
        self._transform = self._build_transform()

    @staticmethod
    def _build_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._checkpoint_path.exists():
            raise RuntimeError(
                f"Checkpoint not found: {self._checkpoint_path}. Train with: python train.py"
            )

        checkpoint = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=False
        )
        self._model_name = checkpoint["config"].get("model", self.config["model"])
        self._class_names = checkpoint.get("class_names", self.config["class_names"])
        self._model = build_model(self._model_name, self.config["num_classes"])
        load_checkpoint(self._checkpoint_path, self._model, self.device)
        self._model.eval()

        image_size = self.config.get("image_size", 224)
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def predict(self, image: Image.Image) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self._model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_idx = probabilities.argmax().item()
        predicted = self._class_names[top_idx]

        return PredictionResult(
            prediction=predicted,
            prediction_label=CLASS_LABELS.get(predicted, predicted),
            confidence=round(probabilities[top_idx].item(), 4),
            probabilities={
                CLASS_LABELS.get(name, name): round(probabilities[i].item(), 4)
                for i, name in enumerate(self._class_names)
            },
            top_idx=top_idx,
        )

    def predict_with_gradcam(self, image: Image.Image) -> tuple[PredictionResult, Image.Image]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        image_rgb = image.convert("RGB")
        result = self.predict(image_rgb)

        target_layer = get_gradcam_target_layer(self._model, self._model_name)
        gradcam = GradCAM(self._model, target_layer)
        try:
            tensor = self._transform(image_rgb).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            heatmap = gradcam.generate(tensor, class_idx=result.top_idx)
            overlay = overlay_heatmap(image_rgb, heatmap)
        finally:
            gradcam.close()

        return result, overlay

    @staticmethod
    def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()

    @staticmethod
    def bytes_to_image(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")


_default_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = InferenceEngine()
    return _default_engine
