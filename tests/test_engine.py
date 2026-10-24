from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from deepweeds.config import ModelConfig, TrainConfig
from deepweeds.data import DeepWeedsData
from deepweeds.engine import Trainer


class MockModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 3 * 224 * 224 = 150528, but we'll use adaptive pool to reduce
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        x = self.pool(x)  # (batch, 3, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 3)
        return self.fc(x)


@dataclass
class MockDataConfig:
    img_size: int = 224


class TestTrainer:
    @pytest.fixture
    def mock_data(self):
        data = MagicMock(spec=DeepWeedsData)
        data.class_weights = torch.ones(9)
        data.class_names = ["c" + str(i) for i in range(9)]
        data.label2id = {f"c{i}": i for i in range(9)}
        data.cfg = MockDataConfig()

        # Create mock loaders
        mock_train_loader = MagicMock()
        mock_train_loader.__iter__ = MagicMock(
            return_value=iter(
                [
                    (torch.randn(4, 3, 224, 224), torch.randint(0, 9, (4,)))
                    for _ in range(3)
                ]
            )
        )
        mock_train_loader.__len__ = MagicMock(return_value=3)

        mock_val_loader = MagicMock()
        mock_val_loader.__iter__ = MagicMock(
            return_value=iter(
                [
                    (torch.randn(4, 3, 224, 224), torch.randint(0, 9, (4,)))
                    for _ in range(2)
                ]
            )
        )

        data.train_loader.return_value = mock_train_loader
        data.val_loader.return_value = mock_val_loader

        return data

    @pytest.fixture
    def trainer(self, mock_data, tmp_path, device):
        model = MockModel()
        model_cfg = ModelConfig(name="test_model", pretrained=False)
        train_cfg = TrainConfig(
            epochs=2,
            lr=1e-4,
            weight_decay=1e-4,
            label_smoothing=0.1,
            balance="none",
            patience=2,
            seed=42,
            amp=False,
            run_root=tmp_path,
            run_name="test",
        )

        trainer = Trainer(model, mock_data, model_cfg, train_cfg, tmp_path, device)
        return trainer

    def test_initialization(self, trainer, tmp_path):
        assert trainer.model is not None
        assert trainer.run_dir == tmp_path
        assert trainer.best_path == tmp_path / "weights" / "best.pt"
        assert trainer.best_f1 == -1.0
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_criterion_weighted_loss(self, mock_data, tmp_path, device):
        model = MockModel()
        model_cfg = ModelConfig(name="test", pretrained=False)
        train_cfg = TrainConfig(balance="weighted_loss", label_smoothing=0.1)

        trainer = Trainer(model, mock_data, model_cfg, train_cfg, tmp_path, device)
        criterion = trainer.criterion

        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert criterion.weight is not None

    def test_build_criterion_none(self, mock_data, tmp_path, device):
        model = MockModel()
        model_cfg = ModelConfig(name="test", pretrained=False)
        train_cfg = TrainConfig(balance="none", label_smoothing=0.1)

        trainer = Trainer(model, mock_data, model_cfg, train_cfg, tmp_path, device)
        criterion = trainer.criterion

        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert criterion.weight is None

    def test_train_one_epoch(self, trainer):
        loss = trainer._train_one_epoch(trainer.data.train_loader())

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate(self, trainer):
        val_loss, y_true, y_prob = trainer._validate(trainer.data.val_loader())

        assert isinstance(val_loss, float)
        assert val_loss >= 0
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_prob, np.ndarray)
        assert y_prob.shape[1] == 9  # num_classes
        assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4)

    def test_predict(self, trainer):
        y_true, y_prob = trainer.predict(trainer.data.val_loader())

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_prob, np.ndarray)
        assert y_prob.shape[1] == 9
        assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-4)

    def test_save_checkpoint(self, trainer, tmp_path):
        # Replace mock data with simple objects that can be pickled
        trainer.data.class_names = ["c0", "c1", "c2"]
        trainer.data.label2id = {"c0": 0, "c1": 1, "c2": 2}
        trainer.data.cfg = MockDataConfig()

        trainer._save_checkpoint(epoch=1, f1=0.85)

        assert trainer.best_path.exists()

        # Verify checkpoint can be loaded
        ckpt = torch.load(trainer.best_path, map_location="cpu", weights_only=True)
        assert "state_dict" in ckpt
        assert "class_names" in ckpt
        assert "config" in ckpt
        assert ckpt["epoch"] == 1
        assert ckpt["val_macro_f1"] == 0.85


class TestTrainerFit:
    @pytest.fixture
    def mock_data_for_fit(self):
        data = MagicMock(spec=DeepWeedsData)
        data.class_weights = torch.ones(9)
        data.class_names = ["c" + str(i) for i in range(9)]
        data.label2id = {f"c{i}": i for i in range(9)}
        data.cfg = MockDataConfig()

        # Create mock loaders that return predictable results
        # Use side_effect to return fresh iterators each time
        train_batches = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) for _ in range(2)
        ]
        val_batches = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) for _ in range(2)
        ]

        def make_train_loader(balance=None):
            loader = MagicMock()
            loader.__iter__ = MagicMock(side_effect=lambda: iter(train_batches))
            loader.__len__ = MagicMock(return_value=2)
            return loader

        def make_val_loader():
            loader = MagicMock()
            loader.__iter__ = MagicMock(side_effect=lambda: iter(val_batches))
            return loader

        data.train_loader = make_train_loader
        data.val_loader = make_val_loader

        return data

    @patch("deepweeds.engine.get_logger")
    def test_fit_runs_epochs(
        self, mock_get_logger, mock_data_for_fit, tmp_path, device
    ):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        model = MockModel()
        model_cfg = ModelConfig(name="test", pretrained=False)
        train_cfg = TrainConfig(
            epochs=3,
            lr=1e-4,
            patience=10,  # High patience to run all epochs
            amp=False,
        )

        trainer = Trainer(
            model, mock_data_for_fit, model_cfg, train_cfg, tmp_path, device
        )
        # Ensure data is picklable for checkpoint saving
        trainer.data.class_names = ["c0", "c1", "c2"]
        trainer.data.label2id = {"c0": 0, "c1": 1, "c2": 2}
        trainer.data.cfg = MockDataConfig()

        best_f1 = trainer.fit()

        assert isinstance(best_f1, float)
        assert best_f1 >= 0
        assert len(trainer.history) == 3
        assert (tmp_path / "history.csv").exists()
        assert (tmp_path / "weights" / "best.pt").exists()

    @patch("deepweeds.engine.get_logger")
    def test_fit_early_stopping(
        self, mock_get_logger, mock_data_for_fit, tmp_path, device
    ):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        model = MockModel()
        model_cfg = ModelConfig(name="test", pretrained=False)
        train_cfg = TrainConfig(
            epochs=10,
            lr=1e-4,
            patience=1,  # Stop after 1 epoch without improvement
            amp=False,
        )

        trainer = Trainer(
            model, mock_data_for_fit, model_cfg, train_cfg, tmp_path, device
        )
        # Ensure data is picklable for checkpoint saving
        trainer.data.class_names = ["c0", "c1", "c2"]
        trainer.data.label2id = {"c0": 0, "c1": 1, "c2": 2}
        trainer.data.cfg = MockDataConfig()

        best_f1 = trainer.fit()

        # Should stop early (less than 10 epochs)
        assert len(trainer.history) < 10
        assert (
            len(trainer.history) >= 2
        )  # At least 2 epochs (first improves, second doesn't)
