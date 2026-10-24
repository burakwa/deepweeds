from unittest.mock import patch

import numpy as np
import pandas as pd

from deepweeds.metrics import MetricsResult
from deepweeds.reports import (
    _save_fig,
    save_all,
    save_class_distribution,
    save_confusion_matrix,
    save_misclassified,
    save_per_class_f1,
    save_training_curves,
)


class TestSaveFig:
    def test_saves_figure(self, tmp_path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        out_path = tmp_path / "test.png"
        _save_fig(fig, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)

    def test_creates_parent_directory(self, tmp_path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])

        out_path = tmp_path / "subdir" / "test.png"
        _save_fig(fig, out_path)

        assert out_path.exists()
        plt.close(fig)


class TestSaveClassDistribution:
    def test_saves_bar_chart(self, tmp_path):
        counts = pd.Series([100, 50, 75], index=["A", "B", "C"])
        class_names = ["A", "B", "C"]

        out_path = tmp_path / "dist.png"
        save_class_distribution(counts, class_names, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestSaveTrainingCurves:
    def test_saves_curves(self, tmp_path):
        history = [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_macro_f1": 0.8,
                "val_accuracy": 0.85,
            },
            {
                "epoch": 2,
                "train_loss": 0.4,
                "val_loss": 0.35,
                "val_macro_f1": 0.85,
                "val_accuracy": 0.88,
            },
        ]

        out_path = tmp_path / "curves.png"
        save_training_curves(history, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestSaveConfusionMatrix:
    def test_saves_heatmap(self, tmp_path):
        cm = [[10, 1, 0], [0, 15, 2], [1, 0, 12]]
        class_names = ["A", "B", "C"]

        out_path = tmp_path / "cm.png"
        save_confusion_matrix(cm, class_names, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestSavePerClassF1:
    def test_saves_horizontal_bar(self, tmp_path):
        per_class = {
            "A": {"f1-score": 0.9, "support": 100},
            "B": {"f1-score": 0.7, "support": 50},
            "C": {"f1-score": 0.8, "support": 75},
        }

        out_path = tmp_path / "per_class.png"
        save_per_class_f1(per_class, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestSaveMisclassified:
    def test_saves_grid_when_errors_exist(self, tmp_path):
        val_df = pd.DataFrame(
            {
                "_path": [str(tmp_path / f"img_{i}.jpg") for i in range(5)],
            }
        )
        # Create dummy images
        from PIL import Image

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            img.save(tmp_path / f"img_{i}.jpg")

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([1, 1, 2, 0, 0])  # Errors at indices 0, 1, 4
        y_prob = np.array(
            [
                [0.4, 0.6, 0.0],
                [0.3, 0.7, 0.0],
                [0.0, 0.0, 1.0],
                [0.9, 0.1, 0.0],
                [0.6, 0.4, 0.0],
            ]
        )
        class_names = ["A", "B", "C"]

        out_path = tmp_path / "misclassified.png"
        save_misclassified(val_df, y_true, y_pred, y_prob, class_names, out_path, k=12)

        assert out_path.exists()

    def test_returns_early_when_no_errors(self, tmp_path):
        val_df = pd.DataFrame({"_path": ["a.jpg"]})
        y_true = np.array([0])
        y_pred = np.array([0])
        y_prob = np.array([[1.0, 0.0]])
        class_names = ["A", "B"]

        out_path = tmp_path / "misclassified.png"
        save_misclassified(val_df, y_true, y_pred, y_prob, class_names, out_path)

        # Should not create file when no errors
        assert not out_path.exists()


class TestSaveAll:
    def test_saves_all_reports(self, tmp_path):
        metrics = MetricsResult(
            macro_f1=0.85,
            accuracy=0.88,
            per_class={
                "A": {"f1-score": 0.9, "precision": 0.9, "recall": 0.9, "support": 10},
                "B": {"f1-score": 0.8, "precision": 0.8, "recall": 0.8, "support": 10},
            },
            confusion=[[9, 1], [1, 9]],
        )
        history = [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_macro_f1": 0.8,
                "val_accuracy": 0.85,
            },
        ]
        class_names = ["A", "B"]
        val_df = pd.DataFrame({"_path": ["a.jpg", "b.jpg"]})
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9]])

        with (
            patch("deepweeds.reports.save_confusion_matrix") as mock_cm,
            patch("deepweeds.reports.save_per_class_f1") as mock_pcf1,
            patch("deepweeds.reports.save_training_curves") as mock_tc,
            patch("deepweeds.reports.save_misclassified") as mock_mis,
        ):
            reports_dir = save_all(
                tmp_path, metrics, history, class_names, val_df, y_true, y_pred, y_prob
            )

            assert reports_dir == tmp_path / "reports"
            assert reports_dir.exists()
            mock_cm.assert_called_once()
            mock_pcf1.assert_called_once()
            mock_tc.assert_called_once()
            mock_mis.assert_called_once()

            # Check metrics.json was created
            assert (tmp_path / "metrics.json").exists()
