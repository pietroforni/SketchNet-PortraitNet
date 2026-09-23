from __future__ import annotations

from pathlib import Path

from .predict import predict_image
from .tasks import TASKS


def classify_artwork(
    image_path: Path,
    sketch_checkpoint: Path,
    portrait_checkpoint: Path,
    device_name: str = "auto",
) -> dict:
    sketch_result = predict_image(
        TASKS["sketchnet"], image_path, sketch_checkpoint, device_name
    )
    stages = {"sketchnet": sketch_result}
    if sketch_result["label"] == "sketch":
        return {"label": "sketch", "stages": stages}

    portrait_result = predict_image(
        TASKS["portraitnet"], image_path, portrait_checkpoint, device_name
    )
    stages["portraitnet"] = portrait_result
    return {"label": portrait_result["label"], "stages": stages}

