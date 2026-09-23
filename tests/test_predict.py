from pathlib import Path

import pytest

from artnet import predict
from artnet.tasks import TASKS


class FakePredictor:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, image_path: Path) -> dict:
        label = "sketch" if "sketch" in image_path.stem else "painting"
        return {"label": label}


def test_sort_folder_copies_images_and_preserves_subfolders(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "collection"
    nested = source / "nested"
    nested.mkdir(parents=True)
    painting = source / "painting.jpg"
    sketch = nested / "sketch.png"
    ignored = source / "notes.txt"
    painting.write_bytes(b"painting")
    sketch.write_bytes(b"sketch")
    ignored.write_text("not an image")
    monkeypatch.setattr(predict, "ImagePredictor", FakePredictor)

    result = predict.sort_folder(
        TASKS["sketchnet"], source, Path("unused.pt"), "cpu"
    )

    assert painting.read_bytes() == b"painting"
    assert sketch.read_bytes() == b"sketch"
    assert (tmp_path / "collection_painting" / "painting.jpg").read_bytes() == b"painting"
    assert (tmp_path / "collection_sketch" / "nested" / "sketch.png").read_bytes() == b"sketch"
    assert result["total"] == 2
    assert result["outputs"]["painting"]["count"] == 1
    assert result["outputs"]["sketch"]["count"] == 1


def test_sort_folder_refuses_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "collection"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"image")
    (tmp_path / "collection_sketch").mkdir()

    with pytest.raises(FileExistsError, match="Output already exists"):
        predict.sort_folder(TASKS["sketchnet"], source, Path("unused.pt"), "cpu")
