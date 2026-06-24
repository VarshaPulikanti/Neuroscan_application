"""FastAPI inference service for brain MRI classification."""

import io
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGENET_MEAN, IMAGENET_STD, load_config
from src.utils import build_model, get_device, load_checkpoint

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

config = load_config()
device = get_device()
checkpoint_path = (
    PROJECT_ROOT / config["output_dir"] / "checkpoints" / config["checkpoint_name"]
)

app = FastAPI(
    title="NeuroScan API",
    description="Brain MRI tumor classification inference API",
    version="1.0.0",
)

_model = None
_class_names: list[str] = []
_model_name = ""


class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]


def _get_transform():
    image_size = config.get("image_size", 224)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_model():
    global _model, _class_names, _model_name
    if _model is not None:
        return

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint_path}. Train with: python train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _model_name = checkpoint["config"].get("model", config["model"])
    _class_names = checkpoint.get("class_names", config["class_names"])
    _model = build_model(_model_name, config["num_classes"])
    load_checkpoint(checkpoint_path, _model, device)
    _model.eval()


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
    except RuntimeError:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with python train.py",
        )

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    transform = _get_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_idx = probabilities.argmax().item()
    predicted = _class_names[top_idx]

    return PredictionResponse(
        prediction=predicted,
        prediction_label=CLASS_LABELS.get(predicted, predicted),
        confidence=round(probabilities[top_idx].item(), 4),
        probabilities={
            CLASS_LABELS.get(name, name): round(probabilities[i].item(), 4)
            for i, name in enumerate(_class_names)
        },
    )
