from artnet.metrics import classification_metrics


def test_classification_metrics() -> None:
    result = classification_metrics(
        [0, 0, 1, 1], [0, 1, 1, 1], ("painting", "sketch")
    )
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["per_class"]["sketch"]["recall"] == 1.0

