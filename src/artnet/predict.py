from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .model import choose_device, evaluation_transform, load_checkpoint
from .tasks import TaskSpec


@torch.inference_mode()
def predict_image(
    task: TaskSpec, image_path: Path, checkpoint: Path, device_name: str = "auto"
) -> dict:
    device = choose_device(device_name)
    model, metadata = load_checkpoint(
        checkpoint,
        device,
        expected_task=task.key,
        legacy_class_names=task.class_names,
    )
    with Image.open(image_path) as image:
        tensor = evaluation_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    index = int(probabilities.argmax().item())
    class_names = metadata.get("class_names", list(task.class_names))
    return {
        "task": task.key,
        "label": class_names[index],
        "confidence": float(probabilities[index].item()),
        "probabilities": {
            name: float(probabilities[position].item()) for position, name in enumerate(class_names)
        },
    }

