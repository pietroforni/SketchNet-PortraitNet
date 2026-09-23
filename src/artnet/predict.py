from __future__ import annotations

import shutil
from pathlib import Path

import torch
from PIL import Image

from .model import choose_device, evaluation_transform, load_checkpoint
from .tasks import TaskSpec

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImagePredictor:
    def __init__(
        self, task: TaskSpec, checkpoint: Path, device_name: str = "auto"
    ) -> None:
        self.task = task
        self.device = choose_device(device_name)
        self.model, metadata = load_checkpoint(
            checkpoint,
            self.device,
            expected_task=task.key,
            legacy_class_names=task.class_names,
        )
        self.class_names = metadata.get("class_names", list(task.class_names))
        self.transform = evaluation_transform()

    @torch.inference_mode()
    def predict(self, image_path: Path) -> dict:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu()
        index = int(probabilities.argmax().item())
        return {
            "task": self.task.key,
            "label": self.class_names[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                name: float(probabilities[position].item())
                for position, name in enumerate(self.class_names)
            },
        }


def predict_image(
    task: TaskSpec, image_path: Path, checkpoint: Path, device_name: str = "auto"
) -> dict:
    return ImagePredictor(task, checkpoint, device_name).predict(image_path)


def sort_folder(
    task: TaskSpec, source: Path, checkpoint: Path, device_name: str = "auto"
) -> dict:
    if not source.is_dir():
        raise ValueError(f"Source is not a directory: {source}")

    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in: {source}")

    output_dirs = {
        label: source.with_name(f"{source.name}_{label}") for label in task.class_names
    }
    existing = [path for path in output_dirs.values() if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {paths}")

    predictor = ImagePredictor(task, checkpoint, device_name)
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True)

    counts = dict.fromkeys(task.class_names, 0)
    for image_path in images:
        result = predictor.predict(image_path)
        label = result["label"]
        destination = output_dirs[label] / image_path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)
        counts[label] += 1

    return {
        "task": task.key,
        "source": str(source),
        "total": len(images),
        "outputs": {
            label: {"directory": str(output_dirs[label]), "count": counts[label]}
            for label in task.class_names
        },
    }
