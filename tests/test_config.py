import pytest
from pathlib import Path
import pandas as pd

from deepweeds.config import (
    CLASS_NAMES, NUM_CLASSES, BalanceStrategy,
    DataConfig, ModelConfig, TrainConfig,
    jsonable, save_json, build_label_map
)


class TestConstants:
    def test_class_names_length(self):
        assert len(CLASS_NAMES) == 9
        assert NUM_CLASSES == 9

    def test_class_names_content(self):
        expected = (
            "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
            "Rubber vine", "Siam weed", "Snake weed", "Negative"
        )
        assert CLASS_NAMES == expected


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.data_dir == Path("data")
        assert cfg.image_dir is None
        assert cfg.image_col is None
        assert cfg.label_col is None
        assert cfg.img_size == 224
        assert cfg.batch_size == 32
        assert cfg.num_workers == 4
        assert cfg.use_aug is True

    def test_custom_values(self):
        cfg = DataConfig(
            data_dir=Path("custom_data"),
            img_size=256,
            batch_size=16,
            num_workers=2,
            use_aug=False,
        )
        assert cfg.data_dir == Path("custom_data")
        assert cfg.img_size == 256
        assert cfg.batch_size == 16
        assert cfg.num_workers == 2
        assert cfg.use_aug is False


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.name == "resnet50"
        assert cfg.num_classes == NUM_CLASSES
        assert cfg.pretrained is True

    def test_custom_values(self):
        cfg = ModelConfig(name="resnet18", num_classes=10, pretrained=False)
        assert cfg.name == "resnet18"
        assert cfg.num_classes == 10
        assert cfg.pretrained is False


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.epochs == 25
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1e-4
        assert cfg.label_smoothing == 0.1
        assert cfg.balance == "weighted_loss"
        assert cfg.patience == 5
        assert cfg.seed == 42
        assert cfg.amp is True
        assert cfg.deterministic is False
        assert cfg.run_root == Path("runs")
        assert cfg.run_name is None

    def test_custom_values(self):
        cfg = TrainConfig(
            epochs=10,
            lr=5e-5,
            balance="balanced_sampler",
            run_name="test_run",
        )
        assert cfg.epochs == 10
        assert cfg.lr == 5e-5
        assert cfg.balance == "balanced_sampler"
        assert cfg.run_name == "test_run"


class TestJsonable:
    def test_path_conversion(self):
        path = Path("test/path")
        result = jsonable(path)
        # On Windows, Path uses backslash separator
        assert result == str(path)

    def test_dataclass_conversion(self):
        cfg = DataConfig(data_dir=Path("data"))
        result = jsonable(cfg)
        assert isinstance(result, dict)
        assert result["data_dir"] == "data"

    def test_dict_conversion(self):
        d = {"key": Path("value")}
        result = jsonable(d)
        assert result == {"key": "value"}

    def test_list_conversion(self):
        lst = [Path("a"), Path("b")]
        result = jsonable(lst)
        assert result == ["a", "b"]

    def test_none_and_primitives(self):
        assert jsonable(None) is None
        assert jsonable(42) == 42
        assert jsonable(3.14) == 3.14
        assert jsonable(True) is True
        assert jsonable("string") == "string"


class TestSaveJson:
    def test_save_and_load(self, tmp_path):
        data = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "test.json"
        save_json(path, data)
        
        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data


class TestBuildLabelMap:
    def test_numeric_labels_already_mapped(self):
        series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8] * 2)
        mapping = build_label_map(series)
        assert mapping == {i: i for i in range(9)}

    def test_string_labels_matched_by_name(self):
        series = pd.Series([CLASS_NAMES[i] for i in range(9)] * 2)
        mapping = build_label_map(series)
        assert all(v is not None for v in mapping.values())
        assert len(set(mapping.values())) == 9

    def test_case_insensitive_matching(self):
        series = pd.Series([name.lower() for name in CLASS_NAMES] * 2)
        mapping = build_label_map(series)
        assert all(v is not None for v in mapping.values())

    def test_raises_on_unmappable_labels(self):
        series = pd.Series(["unknown_class"] * 9)
        with pytest.raises(ValueError, match="eşleşmedi"):
            build_label_map(series)

    def test_raises_on_duplicate_mapping(self):
        series = pd.Series(["Chinee apple"] * 9)
        with pytest.raises(ValueError, match="eşleşmedi"):
            build_label_map(series)