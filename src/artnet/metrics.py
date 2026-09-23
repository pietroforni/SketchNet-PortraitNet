from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def classification_metrics(
    targets: list[int], predictions: list[int], class_names: tuple[str, ...]
) -> dict:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "correct": int(np.equal(targets, predictions).sum()),
        "total": len(targets),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def artist_bootstrap_interval(
    targets: list[int], predictions: list[int], artists: list[str], *, seed: int, samples: int = 10_000
) -> dict[str, list[float] | int]:
    unique_artists = sorted(set(artists))
    indices = {artist: np.flatnonzero(np.asarray(artists) == artist) for artist in unique_artists}
    rng = np.random.default_rng(seed)
    accuracy_values: list[float] = []
    balanced_values: list[float] = []
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)

    for _ in range(samples):
        selected = rng.choice(unique_artists, size=len(unique_artists), replace=True)
        sample_indices = np.concatenate([indices[str(artist)] for artist in selected])
        sampled_targets = target_array[sample_indices]
        sampled_predictions = prediction_array[sample_indices]
        accuracy_values.append(float(np.mean(sampled_targets == sampled_predictions)))
        if len(np.unique(sampled_targets)) == 2:
            balanced_values.append(float(balanced_accuracy_score(sampled_targets, sampled_predictions)))

    return {
        "samples": samples,
        "accuracy_95_ci": np.quantile(accuracy_values, [0.025, 0.975]).tolist(),
        "balanced_accuracy_95_ci": np.quantile(balanced_values, [0.025, 0.975]).tolist(),
    }
