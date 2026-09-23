import csv
from collections import defaultdict
from pathlib import Path

import pytest

from artnet.data import artist_from_filename, discover_records, read_manifest
from artnet.tasks import TASKS


def test_artist_from_filename() -> None:
    assert artist_from_filename("vincent-van-gogh_starry-night.jpg") == "vincent-van-gogh"
    assert TASKS["sketchnet"].class_directories["sketch"] == "sketches"
    assert TASKS["portraitnet"].class_directories["general_painting"] == "general_paintings"


def test_read_manifest_filters_split(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "split"])
        writer.writeheader()
        writer.writerows(
            [
                {"path": "first.jpg", "split": "train"},
                {"path": "second.jpg", "split": "test"},
            ]
        )
    assert read_manifest(path, "test") == [{"path": "second.jpg", "split": "test"}]


@pytest.mark.parametrize("task_name,expected", [("sketchnet", 3343), ("portraitnet", 3077)])
def test_committed_manifests_are_artist_disjoint(task_name: str, expected: int) -> None:
    rows = read_manifest(TASKS[task_name].manifest)
    artist_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        artist_splits[row["artist"]].add(row["split"])
    assert len(rows) == expected
    assert all(len(splits) == 1 for splits in artist_splits.values())


def test_discovery_rejects_a_corrupt_image(tmp_path: Path) -> None:
    (tmp_path / "paintings").mkdir()
    (tmp_path / "sketches").mkdir()
    (tmp_path / "paintings" / "artist_broken.jpg").write_text("not an image")
    with pytest.raises(ValueError, match="Unreadable image"):
        discover_records(tmp_path, TASKS["sketchnet"])
