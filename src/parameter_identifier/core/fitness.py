from __future__ import annotations

import numpy as np


def rmse(expected: np.ndarray, predicted: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if expected.shape != predicted.shape:
        raise ValueError(f"RMSE shape mismatch: expected {expected.shape}, got {predicted.shape}.")
    return float(np.sqrt(np.mean(np.square(expected - predicted))))


def weighted_rmse(
    expected: np.ndarray,
    predicted: np.ndarray,
    skeleton_indices: np.ndarray | None = None,
    skeleton_weight: float = 0.0,
) -> float:
    skeleton_weight = float(skeleton_weight)
    if skeleton_weight < 0 or skeleton_weight > 1:
        raise ValueError("skeleton_weight must be between 0 and 1.")
    curve_error = rmse(expected, predicted)
    if skeleton_weight == 0 or skeleton_indices is None or len(skeleton_indices) == 0:
        return curve_error
    skeleton_error = rmse(expected[skeleton_indices], predicted[skeleton_indices])
    return (1.0 - skeleton_weight) * curve_error + skeleton_weight * skeleton_error
