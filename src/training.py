"""Training and evaluation loops for the character RNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.losses import sequence_cross_entropy_loss
from src.optimizers import adam_update, initialize_adam_state, sgd_update
from src.preprocessing import one_hot_encode
from src.rnn import (
    ParameterDict,
    bptt,
    clip_gradients,
    global_gradient_norm,
    initialize_parameters,
    rnn_forward,
)
from src.utils import iter_sequence_windows, save_parameters


@dataclass
class TrainingHistory:
    """Measured values collected during training."""

    losses: list[float]
    gradient_norms: list[float]


def train_step(
    parameters: ParameterDict,
    inputs: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    learning_rate: float,
    optimizer: str = "sgd",
    adam_state: dict[str, dict[str, np.ndarray]] | None = None,
    adam_timestep: int = 1,
    clip_value: float | None = 5.0,
) -> tuple[float, float, dict[str, dict[str, np.ndarray]] | None]:
    """Run one forward, backward, and update step."""
    input_array = np.asarray(inputs, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    X = one_hot_encode(input_array, parameters["Wax"].shape[1])
    a0 = np.zeros((parameters["Waa"].shape[0], 1))

    hidden_states, predictions = rnn_forward(X, a0, parameters)
    loss = sequence_cross_entropy_loss(predictions, target_array)
    gradients = bptt(X, target_array, parameters, hidden_states, predictions)
    gradient_norm = global_gradient_norm(gradients)

    if clip_value is not None:
        gradients = clip_gradients(gradients, clip_value)

    if optimizer == "sgd":
        sgd_update(parameters, gradients, learning_rate)
    elif optimizer == "adam":
        if adam_state is None:
            adam_state = initialize_adam_state(parameters)
        adam_update(parameters, gradients, adam_state, timestep=adam_timestep, learning_rate=learning_rate)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'.")

    return loss, gradient_norm, adam_state


def train(
    encoded_text: np.ndarray | list[int],
    vocab_size: int,
    hidden_size: int = 64,
    sequence_length: int = 25,
    epochs: int = 10,
    learning_rate: float = 0.01,
    optimizer: str = "adam",
    clip_value: float | None = 5.0,
    seed: int | None = 0,
    checkpoint_path: str | Path | None = None,
) -> tuple[ParameterDict, TrainingHistory]:
    """Train an RNN over repeated windows from encoded text."""
    windows = iter_sequence_windows(encoded_text, sequence_length)
    if not windows:
        raise ValueError("Not enough encoded text to create one training window.")

    parameters = initialize_parameters(vocab_size, hidden_size, vocab_size, seed=seed)
    history = TrainingHistory(losses=[], gradient_norms=[])
    adam_state = initialize_adam_state(parameters) if optimizer == "adam" else None
    timestep = 0

    for _ in range(epochs):
        for inputs, targets in windows:
            timestep += 1
            loss, gradient_norm, adam_state = train_step(
                parameters,
                inputs,
                targets,
                learning_rate=learning_rate,
                optimizer=optimizer,
                adam_state=adam_state,
                adam_timestep=timestep,
                clip_value=clip_value,
            )
            history.losses.append(loss)
            history.gradient_norms.append(gradient_norm)

    if checkpoint_path is not None:
        save_parameters(checkpoint_path, parameters)

    return parameters, history


def evaluate_loss(
    parameters: ParameterDict,
    encoded_text: np.ndarray | list[int],
    sequence_length: int,
) -> float:
    """Average sequence loss across all windows in encoded text."""
    windows = iter_sequence_windows(encoded_text, sequence_length)
    if not windows:
        raise ValueError("Not enough encoded text to create one evaluation window.")

    losses = []
    for inputs, targets in windows:
        X = one_hot_encode(inputs, parameters["Wax"].shape[1])
        a0 = np.zeros((parameters["Waa"].shape[0], 1))
        _, predictions = rnn_forward(X, a0, parameters)
        losses.append(sequence_cross_entropy_loss(predictions, targets))

    return float(np.mean(losses))
