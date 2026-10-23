from __future__ import annotations

import timm
import torch.nn as nn

from deepweeds.config import ModelConfig

SUGGESTED_MODELS = (
    "resnet50",
    "resnet34",
    "convnext_tiny",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "vit_small_patch16_224",
)


def build_model(cfg: ModelConfig) -> nn.Module:
    """timm'den model oluşturur; pretrained=False ise ağ iner ama ağırlık indirmez
    (jüri ortamında offline çıkarım için bunu kullanıyoruz)."""
    return timm.create_model(cfg.name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)


def count_parameters(model: nn.Module) -> int:
    """Eğitilebilir parametre sayısını döner."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)