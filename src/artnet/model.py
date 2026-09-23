from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(
    *, num_classes: int = 2, pretrained: bool = True, freeze_encoder: bool = True
) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(2.0, p=0.3),
            transforms.RandomInvert(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    expected_task: str | None = None,
    legacy_class_names: tuple[str, ...] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "model_state" in payload:
        metadata = {key: value for key, value in payload.items() if key != "model_state"}
        class_names = tuple(metadata["class_names"])
        if expected_task and metadata.get("task") not in (None, expected_task):
            raise ValueError(
                f"Checkpoint task {metadata.get('task')!r} does not match {expected_task!r}"
            )
        model = build_model(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(payload["model_state"])
    else:
        if legacy_class_names is None:
            raise ValueError("A task is required to load a legacy state-dict checkpoint")
        model = build_model(num_classes=len(legacy_class_names), pretrained=False)
        model.load_state_dict(payload)
        metadata = {
            "legacy_checkpoint": True,
            "task": expected_task,
            "class_names": list(legacy_class_names),
        }

    model.to(device)
    model.eval()
    return model, metadata


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
