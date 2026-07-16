"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]
    gradcam_image: str | None = None
    scan_id: int | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    class_names: list[str]
    class_labels: dict[str, str]
    model_loaded: bool
    device: str


class ScanSummary(BaseModel):
    id: int
    filename: str
    prediction_label: str
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ScanDetail(ScanSummary):
    prediction: str
    probabilities: dict[str, float]
    image_thumbnail: str | None = None
    gradcam_image: str | None = None
