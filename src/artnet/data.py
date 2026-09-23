from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import imagehash
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from .model import evaluation_transform, training_transform
from .tasks import TaskSpec

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = ("path", "label", "label_index", "artist", "split", "sha256", "phash")


def artist_from_filename(filename: str) -> str:
    if "_" not in filename:
        raise ValueError(f"Cannot derive artist from filename without an underscore: {filename}")
    return filename.split("_", 1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_records(data_dir: Path, task: TaskSpec) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for label, label_index in task.class_to_index.items():
        class_dir = data_dir / task.class_directories[label]
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    perceptual_hash = str(imagehash.phash(image.convert("RGB")))
            except Exception as exc:
                raise ValueError(f"Unreadable image: {path}") from exc
            records.append(
                {
                    "path": path.as_posix(),
                    "label": label,
                    "label_index": label_index,
                    "artist": artist_from_filename(path.name),
                    "sha256": sha256_file(path),
                    "phash": perceptual_hash,
                }
            )
    return records


def find_duplicate_candidates(
    records: list[dict[str, str | int]], *, max_distance: int = 4
) -> list[dict[str, str | int]]:
    candidates: list[dict[str, str | int]] = []
    hashes = [imagehash.hex_to_hash(str(record["phash"])) for record in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            distance = hashes[left] - hashes[right]
            if distance <= max_distance:
                candidates.append(
                    {
                        "left": str(records[left]["path"]),
                        "right": str(records[right]["path"]),
                        "distance": distance,
                        "exact": int(records[left]["sha256"] == records[right]["sha256"]),
                    }
                )
    return candidates


def assign_splits(records: list[dict[str, str | int]], seed: int) -> None:
    labels = np.asarray([int(record["label_index"]) for record in records])
    groups = np.asarray([str(record["artist"]) for record in records])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_by_index: dict[int, int] = {}
    placeholder = np.zeros(len(records))
    for fold, (_, test_indices) in enumerate(splitter.split(placeholder, labels, groups)):
        for index in test_indices:
            fold_by_index[int(index)] = fold
    if len(fold_by_index) != len(records):
        raise RuntimeError("Not every image was assigned to a fold")

    split_for_fold = {0: "test", 1: "val", 2: "train", 3: "train", 4: "train"}
    for index, record in enumerate(records):
        record["split"] = split_for_fold[fold_by_index[index]]


def write_manifest(records: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_duplicate_candidates(candidates: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("left", "right", "distance", "exact"))
        writer.writeheader()
        writer.writerows(candidates)


def prepare_manifest(
    task: TaskSpec, data_dir: Path, manifest: Path, duplicate_report: Path, seed: int
) -> dict:
    records = discover_records(data_dir, task)
    exact_counts = Counter(str(record["sha256"]) for record in records)
    exact_duplicates = sum(count - 1 for count in exact_counts.values() if count > 1)
    if exact_duplicates:
        raise ValueError(
            f"Found {exact_duplicates} exact duplicate image(s); resolve them before splitting"
        )

    candidates = find_duplicate_candidates(records)
    write_duplicate_candidates(candidates, duplicate_report)
    assign_splits(records, seed)
    write_manifest(records, manifest)
    return summarize_records(records, task.class_names) | {
        "task": task.key,
        "near_duplicate_candidates": len(candidates),
    }


def read_manifest(path: Path, split: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def manifest_checksum(path: Path) -> str:
    return sha256_file(path)


def summarize_records(
    records: Iterable[dict[str, str | int]], class_names: tuple[str, ...]
) -> dict:
    rows = list(records)
    by_split: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row.get("split") == split]
        counts = {
            "images": len(selected),
            "artists": len({str(row["artist"]) for row in selected}),
        }
        counts["classes"] = {
            name: sum(str(row["label"]) == name for row in selected) for name in class_names
        }
        by_split[split] = counts
    return {"total_images": len(rows), "splits": by_split}


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Callable):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label_index"])


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    manifest: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, list[dict[str, str]]]:
    rows = read_manifest(manifest, split)
    transform = training_transform() if split == "train" else evaluation_transform()
    dataset = ManifestDataset(rows, transform)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, rows
