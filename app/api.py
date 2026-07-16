"""FastAPI inference service for brain MRI classification."""

import base64
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.auth import router as auth_router
from app.deps import get_current_user_optional
from app.scan_service import save_scan
from app.scans import router as scans_router
from app.schemas import ModelInfoResponse, PredictionResponse
from src.db.database import get_db, init_db
from src.db.models import User
from src.inference import InferenceEngine, get_engine

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    # Skip loading the model at boot on small hosts (e.g. Render free 512MB).
    # Set LOAD_MODEL_ON_STARTUP=1 to warm-load when you have more RAM.
    if os.getenv("LOAD_MODEL_ON_STARTUP", "0") == "1":
        try:
            get_engine().load()
        except RuntimeError:
            pass
    yield


app = FastAPI(
    title="NeuroScan API",
    description="Brain MRI tumor classification inference API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(scans_router)


def _result_to_response(result, gradcam_b64: str | None = None, scan_id: int | None = None):
    return PredictionResponse(
        prediction=result.prediction,
        prediction_label=result.prediction_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_image=gradcam_b64,
        scan_id=scan_id,
    )


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "device": str(engine.device),
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    engine = get_engine()
    if not engine.is_loaded:
        try:
            engine.load()
        except RuntimeError:
            pass
    from src.inference import CLASS_LABELS

    return ModelInfoResponse(
        model_name=engine.model_name or "unknown",
        class_names=engine.class_names,
        class_labels=CLASS_LABELS,
        model_loaded=engine.is_loaded,
        device=str(engine.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False, description="Include Grad-CAM heatmap overlay"),
    user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    engine = get_engine()
    if not engine.is_loaded:
        try:
            engine.load()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Train the model first with python train.py",
            ) from exc

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image")

    contents = await file.read()
    try:
        image = InferenceEngine.bytes_to_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    gradcam_b64 = None
    if gradcam:
        result, overlay = engine.predict_with_gradcam(image)
        gradcam_b64 = base64.b64encode(
            InferenceEngine.image_to_bytes(overlay)
        ).decode("utf-8")
    else:
        result = engine.predict(image)

    scan_id = None
    if user is not None:
        scan = save_scan(
            db,
            user,
            file.filename or "scan.jpg",
            image,
            result,
            gradcam_b64,
        )
        scan_id = scan.id

    return _result_to_response(result, gradcam_b64, scan_id)


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = FRONTEND_DIST / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
