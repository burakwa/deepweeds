import pytest
import numpy as np

from deepweeds.metrics import compute_metrics, MetricsResult


class TestComputeMetrics:
    def test_perfect_predictions_all_classes(self):
        class_names = [f"class_{i}" for i in range(9)]
        y_true = list(range(9)) * 3  # 27 samples, 3 per class
        y_pred = list(range(9)) * 3
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert isinstance(result, MetricsResult)
        assert result.macro_f1 == 1.0
        assert result.accuracy == 1.0
        assert len(result.confusion) == 9
        assert len(result.confusion[0]) == 9
        assert result.per_class["class_0"]["f1-score"] == 1.0

    def test_absent_classes_count_as_zero(self):
        class_names = [f"class_{i}" for i in range(9)]
        # Only 3 classes present (0, 1, 2), each with 3 samples
        y_true = [0, 1, 2] * 3
        y_pred = [0, 1, 2] * 3
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        # 3 perfect classes + 6 absent classes = 3/9 = 0.333...
        assert abs(result.macro_f1 - 3/9) < 1e-9
        assert result.accuracy == 1.0

    def test_partial_correct_predictions(self):
        class_names = ["A", "B", "C"]
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 0]  # Some errors
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert 0 <= result.macro_f1 <= 1
        assert 0 <= result.accuracy <= 1
        assert result.confusion[0][0] == 1  # True positive for class 0
        assert result.confusion[0][1] == 1  # False negative: 0 predicted as 1

    def test_all_wrong_predictions(self):
        class_names = ["A", "B", "C"]
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]  # All wrong
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert result.macro_f1 == 0.0
        assert result.accuracy == 0.0

    def test_binary_case(self):
        class_names = ["neg", "pos"]
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert isinstance(result, MetricsResult)
        assert 0 <= result.macro_f1 <= 1

    def test_confusion_matrix_shape(self):
        class_names = [f"c{i}" for i in range(5)]
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert len(result.confusion) == 5
        assert all(len(row) == 5 for row in result.confusion)

    def test_per_class_metrics_exist(self):
        class_names = ["A", "B", "C"]
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        for name in class_names:
            assert name in result.per_class
            assert "precision" in result.per_class[name]
            assert "recall" in result.per_class[name]
            assert "f1-score" in result.per_class[name]
            assert "support" in result.per_class[name]

    def test_handles_numpy_arrays(self):
        class_names = ["A", "B", "C"]
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        
        result = compute_metrics(y_true, y_pred, class_names)
        
        assert result.macro_f1 == 1.0