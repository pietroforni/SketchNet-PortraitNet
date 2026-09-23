from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display_name: str
    class_names: tuple[str, str]
    class_directories: dict[str, str]
    data_dir: Path
    manifest: Path
    duplicate_report: Path
    checkpoint: Path
    evaluation_report: Path

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}


TASKS = {
    "sketchnet": TaskSpec(
        key="sketchnet",
        display_name="SketchNet",
        class_names=("painting", "sketch"),
        class_directories={"painting": "paintings", "sketch": "sketches"},
        data_dir=Path("data/private/sketchnet"),
        manifest=Path("data/manifests/sketchnet.csv"),
        duplicate_report=Path("reports/sketchnet-duplicate-candidates.csv"),
        checkpoint=Path("artifacts/checkpoints/sketchnet.pt"),
        evaluation_report=Path("reports/sketchnet-evaluation.json"),
    ),
    "portraitnet": TaskSpec(
        key="portraitnet",
        display_name="PortraitNet",
        class_names=("general_painting", "portrait"),
        class_directories={
            "general_painting": "general_paintings",
            "portrait": "portraits",
        },
        data_dir=Path("data/private/portraitnet"),
        manifest=Path("data/manifests/portraitnet.csv"),
        duplicate_report=Path("reports/portraitnet-duplicate-candidates.csv"),
        checkpoint=Path("artifacts/checkpoints/portraitnet.pt"),
        evaluation_report=Path("reports/portraitnet-evaluation.json"),
    ),
}
