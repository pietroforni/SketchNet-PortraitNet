from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from .data import build_loader, manifest_checksum
from .metrics import artist_bootstrap_interval, classification_metrics
from .model import choose_device, load_checkpoint
from .tasks import TaskSpec


@torch.inference_mode()
def evaluate_model(
    task: TaskSpec,
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    device_name: str = "auto",
) -> dict:
    device = choose_device(device_name)
    model, metadata = load_checkpoint(
        checkpoint,
        device,
        expected_task=task.key,
        legacy_class_names=task.class_names,
    )
    expected_manifest_hash = metadata.get("manifest_sha256")
    actual_manifest_hash = manifest_checksum(manifest)
    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise ValueError("Checkpoint and manifest checksums do not match")

    loader, rows = build_loader(
        manifest, split, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        outputs = model(images.to(device))
        targets.extend(labels.tolist())
        predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(targets, predictions, task.class_names)
    artists = [row["artist"] for row in rows]
    metrics["artist_bootstrap"] = artist_bootstrap_interval(
        targets, predictions, artists, seed=seed
    )
    class_counts = Counter(targets)
    metrics["majority_baseline"] = max(class_counts.values()) / len(targets)
    metrics["artists"] = len(set(artists))

    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "task": task.key,
        "split": split,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "manifest": manifest.as_posix(),
        "manifest_sha256": actual_manifest_hash,
        "checkpoint_metadata": metadata,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
