"""Scan history routes."""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.deps import get_current_user
from app.schemas import ScanDetail, ScanSummary
from src.db.database import get_db
from src.db.models import Scan, User

router = APIRouter(prefix="/scans", tags=["scans"])


@router.get("", response_model=list[ScanSummary])
def list_scans(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    scans = (
        db.query(Scan)
        .filter(Scan.user_id == user.id)
        .order_by(Scan.created_at.desc())
        .all()
    )
    return [ScanSummary.model_validate(s) for s in scans]


@router.get("/{scan_id}", response_model=ScanDetail)
def get_scan(
    scan_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user.id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    return ScanDetail(
        id=scan.id,
        filename=scan.filename,
        prediction_label=scan.prediction_label,
        confidence=scan.confidence,
        created_at=scan.created_at,
        prediction=scan.prediction,
        probabilities=json.loads(scan.probabilities),
        image_thumbnail=scan.image_thumbnail,
        gradcam_image=scan.gradcam_image,
    )


@router.delete("/{scan_id}", status_code=204)
def delete_scan(
    scan_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user.id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    db.delete(scan)
    db.commit()
