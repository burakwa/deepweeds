import torch

from deepweeds.config import ModelConfig
from deepweeds.metrics import compute_metrics
from deepweeds.models import build_model


def test_model_forward():
    model = build_model(ModelConfig(name="resnet18", pretrained=False))
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 9)


def test_metrics_perfect_when_all_classes_present():
    """9 sınıfın hepsinde örnek varsa macro-F1 = 1.0 olmalı."""
    names = [f"c{i}" for i in range(9)]
    y = list(range(9)) * 3
    m = compute_metrics(y, y, names)
    assert m.macro_f1 == 1.0
    assert m.accuracy == 1.0
    assert len(m.confusion) == 9 and len(m.confusion[0]) == 9


def test_macro_f1_counts_absent_classes_as_zero():
    """Örneği hiç olmayan sınıflar F1=0 sayılır: 3 mükemmel + 6 boş → 3/9 ≈ 0.333.
    Bu, macro-F1'in sınıf eksikliğini neden bu kadar sert cezalandırdığını gösterir."""
    names = [f"c{i}" for i in range(9)]
    y = [0, 1, 2] * 3
    m = compute_metrics(y, y, names)
    assert abs(m.macro_f1 - 3 / 9) < 1e-9
    assert m.accuracy == 1.0  # doğruluk yine kusursuz — ikisinin farkı tam burada