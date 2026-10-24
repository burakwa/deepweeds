from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepweeds.config import CLASS_NAMES, NUM_CLASSES, DataConfig
from deepweeds.data import (
    DeepWeedsData,
    DeepWeedsDataset,
    build_eval_transform,
    build_train_transform,
    build_tta_transforms,
)


class TestTransforms:
    def test_build_eval_transform(self):
        transform = build_eval_transform(224)
        assert callable(transform)
        # Test on tensor
        img = torch.randn(3, 256, 256)
        from PIL import Image

        pil_img = Image.fromarray(
            (torch.randn(256, 256, 3) * 127 + 127).clamp(0, 255).byte().numpy()
        )
        result = transform(pil_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_build_train_transform_default(self):
        transform = build_train_transform(224, use_aug=True)
        assert callable(transform)
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        result = transform(pil_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_build_train_transform_no_aug(self):
        transform = build_train_transform(224, use_aug=False)
        assert callable(transform)
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        result = transform(pil_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_build_train_transform_strong_aug(self):
        transform = build_train_transform(224, use_aug=True, strong_aug=True)
        assert callable(transform)
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        result = transform(pil_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_build_tta_transforms(self):
        transforms = build_tta_transforms(224)
        assert isinstance(transforms, list)
        assert len(transforms) == 5
        for t in transforms:
            assert callable(t)
            pil_img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            result = t(pil_img)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)


class TestDeepWeedsDataset:
    def test_dataset_length(self, sample_dataframe, mock_images):
        transform = build_eval_transform(224)
        df = sample_dataframe.copy()
        df["_path"] = df["path"].apply(lambda x: str(mock_images / Path(x).name))
        df["_label"] = df["label"]

        dataset = DeepWeedsDataset(df, transform)
        assert len(dataset) == len(df)

    def test_dataset_getitem(self, sample_dataframe, mock_images):
        transform = build_eval_transform(224)
        df = sample_dataframe.copy()
        df["_path"] = df["path"].apply(lambda x: str(mock_images / Path(x).name))
        df["_label"] = df["label"]

        dataset = DeepWeedsDataset(df, transform)
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < NUM_CLASSES


class TestDeepWeedsData:
    def test_init_with_mock_data(self, temp_dir, sample_dataframe, mock_images):
        # Create CSV files
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()
        test_df = sample_dataframe.iloc[90:].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Update paths to point to mock images
        for df in [train_df, val_df, test_df]:
            df["path"] = df["path"].apply(lambda x: str(mock_images / Path(x).name))

        cfg = DataConfig(data_dir=data_dir, image_dir=mock_images)
        data = DeepWeedsData(cfg)

        assert len(data.train_df) == 80
        assert len(data.val_df) == 10
        assert len(data.test_df) == 10
        assert data.class_names == list(CLASS_NAMES)
        assert len(data.class_counts) == NUM_CLASSES

    def test_class_weights_computed(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)

        cfg = DataConfig(data_dir=data_dir, image_dir=mock_images)
        data = DeepWeedsData(cfg)

        assert hasattr(data, "class_weights")
        assert isinstance(data.class_weights, torch.Tensor)
        assert data.class_weights.shape == (NUM_CLASSES,)
        assert (data.class_weights > 0).all()

    def test_sample_weights_property(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)

        cfg = DataConfig(data_dir=data_dir, image_dir=mock_images)
        data = DeepWeedsData(cfg)

        weights = data.sample_weights
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(data.train_df)
        assert (weights > 0).all()

    def test_train_loader(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)

        cfg = DataConfig(
            data_dir=data_dir, image_dir=mock_images, batch_size=4, num_workers=0
        )
        data = DeepWeedsData(cfg)

        loader = data.train_loader(balance="none")
        assert isinstance(loader, torch.utils.data.DataLoader)

        batch = next(iter(loader))
        images, labels = batch
        assert images.shape[0] <= 4
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] <= 4

    def test_val_loader(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)

        cfg = DataConfig(
            data_dir=data_dir, image_dir=mock_images, batch_size=4, num_workers=0
        )
        data = DeepWeedsData(cfg)

        loader = data.val_loader()
        batch = next(iter(loader))
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)

    def test_test_loader(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()
        test_df = sample_dataframe.iloc[90:].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        cfg = DataConfig(
            data_dir=data_dir, image_dir=mock_images, batch_size=4, num_workers=0
        )
        data = DeepWeedsData(cfg)

        loader = data.test_loader()
        batch = next(iter(loader))
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)
        # Test labels should be -1
        assert (labels == -1).all()

    def test_test_loader_raises_without_test_csv(
        self, temp_dir, sample_dataframe, mock_images
    ):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        # No test.csv

        cfg = DataConfig(data_dir=data_dir, image_dir=mock_images)
        data = DeepWeedsData(cfg)

        with pytest.raises(FileNotFoundError, match="test.csv yok"):
            data.test_loader()

    def test_id_series(self, temp_dir, sample_dataframe, mock_images):
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        train_df = sample_dataframe.iloc[:80].copy()
        val_df = sample_dataframe.iloc[80:90].copy()

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)

        cfg = DataConfig(data_dir=data_dir, image_dir=mock_images)
        data = DeepWeedsData(cfg)

        ids = data.id_series(data.val_df)
        assert len(ids) == len(data.val_df)
        assert all(isinstance(x, str) for x in ids)
