from pathlib import Path

import pytest

from artnet import cascade


def test_cascade_stops_after_sketch(monkeypatch) -> None:
    calls = []

    def fake_predict(task, image, checkpoint, device):
        calls.append(task.key)
        return {"task": task.key, "label": "sketch", "confidence": 0.9}

    monkeypatch.setattr(cascade, "predict_image", fake_predict)
    result = cascade.classify_artwork(Path("image.jpg"), Path("sketch.pt"), Path("portrait.pt"))
    assert result["label"] == "sketch"
    assert calls == ["sketchnet"]


@pytest.mark.parametrize("portrait_label", ["portrait", "general_painting"])
def test_cascade_routes_paintings_to_portraitnet(monkeypatch, portrait_label: str) -> None:
    def fake_predict(task, image, checkpoint, device):
        label = "painting" if task.key == "sketchnet" else portrait_label
        return {"task": task.key, "label": label, "confidence": 0.9}

    monkeypatch.setattr(cascade, "predict_image", fake_predict)
    result = cascade.classify_artwork(Path("image.jpg"), Path("sketch.pt"), Path("portrait.pt"))
    assert result["label"] == portrait_label
    assert set(result["stages"]) == {"sketchnet", "portraitnet"}
