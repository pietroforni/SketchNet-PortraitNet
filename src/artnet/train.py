from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import build_loader, manifest_checksum
from .metrics import classification_metrics
from .model import IMAGENET_MEAN, IMAGENET_STD, build_model, choose_device
from .tasks import TaskSpec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    return total_loss / len(targets), targets, predictions


def train_model(
    task: TaskSpec,
    manifest: Path,
    output: Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    device_name: str = "auto",
) -> dict:
    seed_everything(seed)
    device = choose_device(device_name)
    train_loader, train_rows = build_loader(
        manifest, "train", batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    val_loader, val_rows = build_loader(
        manifest, "val", batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    model = build_model(num_classes=len(task.class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    best_balanced_accuracy = -1.0
    history: list[dict] = []

    print(f"Using device: {device}")
    print(f"Training images: {len(train_rows)} | Validation images: {len(val_rows)}")
    for epoch in range(1, epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_targets, val_predictions = run_epoch(model, val_loader, criterion, device)
        train_metrics = classification_metrics(
            train_targets, train_predictions, task.class_names
        )
        val_metrics = classification_metrics(val_targets, val_predictions, task.class_names)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train acc {train_metrics['accuracy']:.3f} | "
            f"val acc {val_metrics['accuracy']:.3f} | "
            f"val balanced {val_metrics['balanced_accuracy']:.3f}"
        )
        if val_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "task": task.key,
                    "class_names": list(task.class_names),
                    "seed": seed,
                    "epoch": epoch,
                    "manifest_sha256": manifest_checksum(manifest),
                    "image_size": 224,
                    "preprocessing": {
                        "resize": 256,
                        "center_crop": 224,
                        "normalization_mean": IMAGENET_MEAN,
                        "normalization_std": IMAGENET_STD,
                    },
                    "validation_metrics": val_metrics,
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "optimizer": "Adam",
                        "loss": "CrossEntropyLoss",
                        "pretrained_weights": "ResNet50_Weights.IMAGENET1K_V1",
                        "device": str(device),
                    },
                },
                output,
            )

    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "checkpoint": str(output),
        "history": str(history_path),
        "best_validation_balanced_accuracy": best_balanced_accuracy,
    }
