"""Scan history persistence helpers."""

from __future__ import annotations

import base64
import io
import json

from PIL import Image

from src.db.models import Scan, User
from src.inference import PredictionResult


def _thumbnail_b64(image: Image.Image, size: int = 200) -> str:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_scan(
    db,
    user: User,
    filename: str,
    image: Image.Image,
    result: PredictionResult,
    gradcam_b64: str | None = None,
) -> Scan:
    scan = Scan(
        user_id=user.id,
        filename=filename,
        prediction=result.prediction,
        prediction_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=json.dumps(result.probabilities),
        image_thumbnail=_thumbnail_b64(image),
        gradcam_image=gradcam_b64,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan
