from pathlib import Path

import pytest
import torch

from artnet import model


def test_checkpoint_task_mismatch_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        model.torch,
        "load",
        lambda *args, **kwargs: {
            "model_state": {},
            "task": "portraitnet",
            "class_names": ["general_painting", "portrait"],
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        model.load_checkpoint(
            Path("unused.pt"),
            torch.device("cpu"),
            expected_task="sketchnet",
        )
