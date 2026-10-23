import torch

from deepweeds.config import ModelConfig
from deepweeds.metrics import compute_metrics
from deepweeds.models import build_model


def test_model_forward():
    model = build_model(ModelConfig(name="resnet18", pretrained=False))
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 9)


def test_metrics():
    names = [f"c{i}" for i in range(9)]
    y = [0, 1, 2] * 3
    m = compute_metrics(y, y, names)
    assert m.macro_f1 == 1.0