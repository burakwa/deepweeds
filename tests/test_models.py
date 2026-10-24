import pytest
from torch import nn

from deepweeds.config import ModelConfig
from deepweeds.models import SUGGESTED_MODELS, build_model, count_parameters


class TestBuildModel:
    @pytest.mark.parametrize(
        "model_name",
        [
            "resnet18",
            "resnet34",
            "resnet50",
        ],
    )
    def test_build_model_creates_module(self, model_name):
        cfg = ModelConfig(name=model_name, pretrained=False)
        model = build_model(cfg)
        assert isinstance(model, nn.Module)

    def test_build_model_correct_num_classes(self):
        cfg = ModelConfig(name="resnet18", num_classes=9, pretrained=False)
        model = build_model(cfg)
        # Check final layer output size
        if hasattr(model, "fc"):
            assert model.fc.out_features == 9
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                assert model.classifier.out_features == 9
            else:
                # For models with Sequential classifier
                last_layer = list(model.classifier.children())[-1]
                assert last_layer.out_features == 9
        elif hasattr(model, "head"):
            if isinstance(model.head, nn.Linear):
                assert model.head.out_features == 9
            else:
                last_layer = list(model.head.children())[-1]
                assert last_layer.out_features == 9

    def test_build_model_pretrained_false_no_weights(self):
        cfg = ModelConfig(name="resnet18", pretrained=False)
        model = build_model(cfg)
        # Should not crash and should create model
        assert model is not None


class TestCountParameters:
    def test_counts_parameters(self):
        cfg = ModelConfig(name="resnet18", pretrained=False)
        model = build_model(cfg)
        count = count_parameters(model)
        assert isinstance(count, int)
        assert count > 0

    def test_only_counts_trainable(self):
        cfg = ModelConfig(name="resnet18", pretrained=False)
        model = build_model(cfg)
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False
        count = count_parameters(model)
        assert count == 0


class TestSuggestedModels:
    def test_list_not_empty(self):
        assert len(SUGGESTED_MODELS) > 0

    def test_all_are_strings(self):
        for m in SUGGESTED_MODELS:
            assert isinstance(m, str)

    def test_contains_common_models(self):
        assert "resnet50" in SUGGESTED_MODELS
        assert "convnext_tiny" in SUGGESTED_MODELS
