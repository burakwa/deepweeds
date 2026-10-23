import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import pandas as pd
import numpy as np
import sys

from deepweeds.cli import (
    _data_cfg,
    _add_data_args,
    cmd_train,
    cmd_predict,
    cmd_evaluate,
    cmd_api,
    main,
)
from deepweeds.config import CLASS_NAMES


class TestDataCfg:
    def test_creates_dataconfig_from_args(self):
        args = MagicMock()
        args.data = "custom_data"
        args.image_dir = "images"
        args.image_col = "img_path"
        args.label_col = "target"
        args.img_size = 256
        args.batch_size = 16
        args.workers = 2
        
        cfg = _data_cfg(args)
        
        assert cfg.data_dir == Path("custom_data")
        assert cfg.image_dir == Path("images")
        assert cfg.image_col == "img_path"
        assert cfg.label_col == "target"
        assert cfg.img_size == 256
        assert cfg.batch_size == 16
        assert cfg.num_workers == 2


class TestAddDataArgs:
    def test_adds_all_arguments(self):
        import argparse
        parser = argparse.ArgumentParser()
        _add_data_args(parser)
        
        # Check all expected arguments are added
        args = parser.parse_args([
            "--data", "d",
            "--image-dir", "img",
            "--image-col", "path",
            "--label-col", "label",
            "--img-size", "224",
            "--batch-size", "32",
            "--workers", "4",
        ])
        assert args.data == "d"
        assert args.image_dir == "img"
        assert args.image_col == "path"
        assert args.label_col == "label"
        assert args.img_size == 224
        assert args.batch_size == 32
        assert args.workers == 4


class TestCmdTrain:
    @patch("deepweeds.cli.DeepWeedsData")
    @patch("deepweeds.cli.build_model")
    @patch("deepweeds.cli.Trainer")
    @patch("deepweeds.cli.Predictor")
    @patch("deepweeds.cli.set_seed")
    @patch("deepweeds.cli.get_device")
    @patch("deepweeds.cli.get_logger")
    @patch("deepweeds.cli.save_json")
    @patch("deepweeds.cli.save_class_distribution")
    def test_cmd_train_runs(
        self,
        mock_save_dist,
        mock_save_json,
        mock_get_logger,
        mock_get_device,
        mock_set_seed,
        mock_predictor_class,
        mock_trainer_class,
        mock_build_model,
        mock_data_class,
        tmp_path,
    ):
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_device.return_value = "cpu"
        
        mock_data = MagicMock()
        mock_data.class_counts = pd.Series([10]*9, index=range(9))
        mock_data.class_names = ["c"+str(i) for i in range(9)]
        mock_data.val_df = pd.DataFrame({"_path": [f"a{i}.jpg" for i in range(5)], "_label": [0,1,2,3,4]})
        mock_data.id_series.return_value = pd.Series([f"id_{i}" for i in range(5)])
        mock_data_class.return_value = mock_data
        
        mock_model = MagicMock()
        mock_build_model.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.fit.return_value = 0.85
        mock_trainer.best_path = tmp_path / "weights" / "best.pt"
        mock_trainer.history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_macro_f1": 0.8, "val_accuracy": 0.85, "lr": 1e-4, "seconds": 10}]
        mock_trainer_class.return_value = mock_trainer
        
        mock_predictor = MagicMock()
        # 9 classes, 5 samples
        y_prob = np.zeros((5, 9))
        y_prob[np.arange(5), [0,1,2,3,4]] = 1.0
        mock_predictor.predict_df.return_value = MagicMock(
            y_true=np.array([0,1,2,3,4]),
            y_prob=y_prob
        )
        mock_predictor.benchmark_latency.return_value = {"mean_ms": 10, "p50_ms": 9, "p95_ms": 12}
        mock_predictor.checkpoint_path = tmp_path / "weights" / "best.pt"
        mock_predictor.size_bytes = 1000
        mock_predictor_class.from_checkpoint.return_value = mock_predictor
        
        mock_save_json.return_value = None
        mock_save_dist.return_value = None
        
        args = MagicMock()
        args.data = "data"
        args.image_dir = None
        args.image_col = None
        args.label_col = None
        args.img_size = 224
        args.batch_size = 32
        args.workers = 4
        args.model = "resnet18"
        args.epochs = 1
        args.lr = 1e-4
        args.wd = 1e-4
        args.label_smoothing = 0.1
        args.balance = "weighted_loss"
        args.patience = 5
        args.seed = 42
        args.run_root = "runs"
        args.run_name = "test_run"
        args.aug = True
        args.amp = True
        args.deterministic = False
        
        result = cmd_train(args)
        
        assert result == 0
        mock_trainer.fit.assert_called_once()


class TestCmdPredict:
    @patch("deepweeds.cli.DeepWeedsData")
    @patch("deepweeds.cli.Predictor")
    @patch("deepweeds.cli.write_predictions_csv")
    @patch("deepweeds.cli.get_logger")
    @patch("deepweeds.cli.compute_metrics")
    def test_cmd_predict_val(
        self,
        mock_compute_metrics,
        mock_get_logger,
        mock_write_csv,
        mock_predictor_class,
        mock_data_class,
        tmp_path,
    ):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_data = MagicMock()
        mock_data.val_df = pd.DataFrame({"_path": ["a.jpg"], "_label": [0]})
        mock_data.id_series.return_value = pd.Series(["a"])
        mock_data.class_names = list(CLASS_NAMES)
        mock_data_class.return_value = mock_data
        
        mock_predictor = MagicMock()
        y_prob = np.zeros((1, 9))
        y_prob[0, 0] = 1.0
        mock_predictor.predict_df.return_value = MagicMock(
            y_true=np.array([0]),
            y_prob=y_prob
        )
        mock_predictor.benchmark_latency.return_value = {"mean_ms": 10}
        mock_predictor.size_bytes = 1000
        mock_predictor.checkpoint_path = tmp_path / "best.pt"
        mock_predictor_class.from_checkpoint.return_value = mock_predictor
        
        mock_write_csv.return_value = pd.DataFrame()
        
        mock_metrics = MagicMock()
        mock_metrics.macro_f1 = 1.0
        mock_compute_metrics.return_value = mock_metrics
        
        args = MagicMock()
        args.data = "data"
        args.image_dir = None
        args.image_col = None
        args.label_col = None
        args.img_size = 224
        args.batch_size = 32
        args.workers = 4
        args.run = str(tmp_path / "best.pt")
        args.split = "val"
        args.output = str(tmp_path / "pred.csv")
        args.calibration = None
        
        result = cmd_predict(args)
        
        assert result == 0
        mock_write_csv.assert_called_once()
        mock_compute_metrics.assert_called_once()


class TestCmdEvaluate:
    @patch("deepweeds.cli.compute_metrics")
    @patch("deepweeds.reports.save_confusion_matrix")
    @patch("deepweeds.cli.save_json")
    @patch("deepweeds.cli.get_logger")
    @patch("pandas.read_csv")
    @patch("deepweeds.cli.build_label_map")
    def test_cmd_evaluate(
        self,
        mock_build_label_map,
        mock_read_csv,
        mock_get_logger,
        mock_save_json,
        mock_save_cm,
        mock_compute_metrics,
        tmp_path,
    ):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock truth and predictions CSVs with string labels matching CLASS_NAMES
        truth_df = pd.DataFrame({"id": ["a", "b"], "label": [CLASS_NAMES[0], CLASS_NAMES[1]]})
        pred_df = pd.DataFrame({"id": ["a", "b"], "prediction": [0, 1]})
        mock_read_csv.side_effect = [truth_df, pred_df]
        
        # Mock build_label_map to return proper mapping
        mock_build_label_map.return_value = {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 1}
        
        mock_metrics = MagicMock()
        mock_metrics.macro_f1 = 1.0
        mock_metrics.accuracy = 1.0
        mock_metrics.per_class = {}
        mock_metrics.confusion = [[1,0],[0,1]]
        mock_compute_metrics.return_value = mock_metrics
        
        args = MagicMock()
        args.truth = "truth.csv"
        args.predictions = "pred.csv"
        args.out_dir = str(tmp_path / "eval")
        
        result = cmd_evaluate(args)
        
        assert result == 0
        mock_compute_metrics.assert_called_once()
        mock_save_json.assert_called_once()


class TestCmdApi:
    @patch("uvicorn.run")
    @patch("deepweeds.api.create_app")
    def test_cmd_api(self, mock_create_app, mock_uvicorn_run):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        args = MagicMock()
        args.model = "model.pt"
        args.host = "127.0.0.1"
        args.port = 8000
        args.device = None
        
        result = cmd_api(args)
        
        assert result == 0
        mock_create_app.assert_called_once_with("model.pt", device=None)
        mock_uvicorn_run.assert_called_once()


class TestMain:
    @patch("deepweeds.cli.cmd_train")
    def test_main_train(self, mock_cmd_train):
        mock_cmd_train.return_value = 0
        
        result = main(["train", "--model", "resnet18"])
        
        assert result == 0
        mock_cmd_train.assert_called_once()

    @patch("deepweeds.cli.cmd_predict")
    def test_main_predict(self, mock_cmd_predict):
        mock_cmd_predict.return_value = 0
        
        result = main(["predict", "--run", "model.pt"])
        
        assert result == 0
        mock_cmd_predict.assert_called_once()