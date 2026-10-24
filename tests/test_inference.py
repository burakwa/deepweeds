
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from deepweeds.config import ModelConfig
from deepweeds.inference import Predictor, PredictResult, write_predictions_csv
from deepweeds.models import build_model


class TestPredictResult:
    def test_named_tuple_structure(self):
        y_true = np.array([0, 1, 2])
        y_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

        result = PredictResult(y_true=y_true, y_prob=y_prob)

        assert result.y_true is y_true
        assert result.y_prob is y_prob
        assert result.y_true.shape == (3,)
        assert result.y_prob.shape == (3, 3)


class TestPredictor:
    @pytest.fixture
    def mock_checkpoint(self, temp_dir):
        """Create a mock checkpoint for testing."""
        model_cfg = ModelConfig(name="resnet18", num_classes=3, pretrained=False)
        model = build_model(model_cfg)

        ckpt = {
            "state_dict": model.state_dict(),
            "class_names": ["A", "B", "C"],
            "label2id": {"A": 0, "B": 1, "C": 2},
            "config": {
                "model": {"name": "resnet18", "num_classes": 3, "pretrained": False},
                "train": {},
                "data": {"img_size": 224},
            },
            "epoch": 1,
            "val_macro_f1": 0.9,
        }

        ckpt_path = temp_dir / "mock_best.pt"
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    def test_from_checkpoint_loads_model(self, mock_checkpoint, device):
        predictor = Predictor.from_checkpoint(mock_checkpoint, device)

        assert predictor.model is not None
        assert predictor.class_names == ["A", "B", "C"]
        assert predictor.img_size == 224
        assert predictor.device == device

    def test_predict_pil_returns_probabilities(self, mock_checkpoint, device):
        predictor = Predictor.from_checkpoint(mock_checkpoint, device)

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        probs = predictor.predict_pil(img)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_df_returns_predict_result(self, mock_checkpoint, device, temp_dir):
        predictor = Predictor.from_checkpoint(mock_checkpoint, device)

        # Create mock dataframe
        img_dir = temp_dir / "imgs"
        img_dir.mkdir()

        paths = []
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            p = img_dir / f"img_{i}.jpg"
            img.save(p)
            paths.append(str(p))

        df = pd.DataFrame({"_path": paths, "_label": [-1] * 5})

        result = predictor.predict_df(df, batch_size=2)

        assert isinstance(result, PredictResult)
        assert result.y_true.shape == (5,)
        assert result.y_prob.shape == (5, 3)
        assert (result.y_true == -1).all()

    def test_benchmark_latency(self, mock_checkpoint, device, temp_dir):
        predictor = Predictor.from_checkpoint(mock_checkpoint, device)

        img_dir = temp_dir / "imgs"
        img_dir.mkdir()

        paths = []
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            p = img_dir / f"img_{i}.jpg"
            img.save(p)
            paths.append(str(p))

        df = pd.DataFrame({"_path": paths, "_label": [-1] * 10})

        latency = predictor.benchmark_latency(df, n=5)

        assert "mean_ms" in latency
        assert "p50_ms" in latency
        assert "p95_ms" in latency
        assert all(v > 0 for v in latency.values())


class TestWritePredictionsCsv:
    def test_writes_csv_with_correct_format(self, temp_dir):
        ids = pd.Series(["img1", "img2", "img3"])
        probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
            ]
        )

        out_path = temp_dir / "predictions.csv"
        df = write_predictions_csv(
            out_path, ids, probs, latency_ms=10.5, size_bytes=1000
        )

        assert out_path.exists()
        assert len(df) == 3
        assert list(df.columns) == [
            "id",
            "prediction",
            "prob_0",
            "prob_1",
            "prob_2",
            "latency_ms",
            "bytes",
        ]
        assert df["latency_ms"].iloc[0] == 10.5
        assert df["bytes"].iloc[0] == 1000
        assert df["id"].is_unique

    def test_predictions_sum_to_one(self, temp_dir):
        ids = pd.Series(["a", "b"])
        probs = np.array([[0.6, 0.4], [0.3, 0.7]])

        out_path = temp_dir / "predictions.csv"
        df = write_predictions_csv(out_path, ids, probs, latency_ms=5.0, size_bytes=500)

        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        row_sums = df[prob_cols].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4)

    def test_with_multipliers(self, temp_dir):
        ids = pd.Series(["a", "b"])
        probs = np.array(
            [
                [0.4, 0.6],  # Would predict class 1
                [0.7, 0.3],  # Would predict class 0
            ]
        )
        multipliers = np.array([2.0, 0.5])  # Boost class 0, reduce class 1

        out_path = temp_dir / "predictions.csv"
        df = write_predictions_csv(
            out_path,
            ids,
            probs,
            latency_ms=5.0,
            size_bytes=500,
            multipliers=multipliers,
        )

        # With multipliers: [0.8, 0.3] -> class 0, [1.4, 0.15] -> class 0
        assert df["prediction"].tolist() == [0, 0]
