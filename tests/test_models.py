"""Tests for NeuroScan utilities."""

import torch
import torch.nn as nn

from src.gradcam import get_gradcam_target_layer
from src.utils import build_model


def test_build_efficientnet():
    model = build_model("efficientnet_b0", num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_build_resnet18():
    model = build_model("resnet18", num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_gradcam_target_layers():
    for name in ("efficientnet_b0", "resnet18"):
        model = build_model(name, num_classes=4)
        layer = get_gradcam_target_layer(model, name)
        assert isinstance(layer, nn.Module)
