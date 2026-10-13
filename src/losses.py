"""Loss functions for character-level sequence prediction."""

from __future__ import annotations

import numpy as np

from src.rnn import stable_softmax


def cross_entropy_loss(y_hat: np.ndarray, target_idx: int, epsilon: float = 1e-12) -> float:
    """Return -log P(target_idx) for one time step."""
    if y_hat.ndim != 2 or y_hat.shape[1] != 1:
        raise ValueError("y_hat must have shape (n_y, 1).")
    if target_idx < 0 or target_idx >= y_hat.shape[0]:
        raise ValueError("target_idx is outside the output vocabulary range.")

    probability = float(y_hat[target_idx, 0])
    return -float(np.log(probability + epsilon))


def sequence_cross_entropy_loss(
    predictions: dict[int, np.ndarray], targets: np.ndarray | list[int]
) -> float:
    """Sum cross-entropy loss over a sequence."""
    target_array = np.asarray(targets, dtype=np.int64)
    if len(predictions) != target_array.size:
        raise ValueError("Number of predictions must match number of targets.")

    return sum(cross_entropy_loss(predictions[t], int(target_array[t])) for t in range(target_array.size))


__all__ = ["cross_entropy_loss", "sequence_cross_entropy_loss", "stable_softmax"]
