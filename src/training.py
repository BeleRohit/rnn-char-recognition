"""Training and evaluation loops for the character RNN."""

from __future__ import annotations

import time
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
    loss_per_chars: list[float]
    gradient_norms: list[float]
    validation_losses: list[float]
    validation_loss_per_chars: list[float]


def train_step(
    parameters: ParameterDict,
    inputs: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    learning_rate: float,
    optimizer: str = "sgd",
    adam_state: dict[str, dict[str, np.ndarray]] | None = None,
    adam_timestep: int = 1,
    clip_value: float | None = 5.0,
    initial_hidden_state: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, dict[str, dict[str, np.ndarray]] | None]:
    """Run one forward, backward, and update step."""
    input_array = np.asarray(inputs, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    X = one_hot_encode(input_array, parameters["Wax"].shape[1])
    if initial_hidden_state is None:
        a0 = np.zeros((parameters["Waa"].shape[0], 1))
    else:
        a0 = initial_hidden_state

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

    return loss, gradient_norm, hidden_states[X.shape[1] - 1], adam_state


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
    progress_every: int | None = None,
    initial_parameters: ParameterDict | None = None,
    validation_data: np.ndarray | list[int] | None = None,
    validation_every_epochs: int = 1,
    stateful: bool = True,
    learning_rate_decay: float = 1.0,
) -> tuple[ParameterDict, TrainingHistory]:
    """Train an RNN over repeated windows from encoded text."""
    windows = iter_sequence_windows(encoded_text, sequence_length)
    if not windows:
        raise ValueError("Not enough encoded text to create one training window.")
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every must be positive when provided.")
    if validation_every_epochs <= 0:
        raise ValueError("validation_every_epochs must be positive.")
    if learning_rate_decay <= 0:
        raise ValueError("learning_rate_decay must be positive.")

    if initial_parameters is None:
        parameters = initialize_parameters(vocab_size, hidden_size, vocab_size, seed=seed)
    else:
        parameters = initial_parameters
        if parameters["Wax"].shape != (hidden_size, vocab_size):
            raise ValueError("initial_parameters shape does not match hidden_size and vocab_size.")
        if parameters["Wya"].shape[0] != vocab_size:
            raise ValueError("initial_parameters output size does not match vocab_size.")

    history = TrainingHistory(
        losses=[],
        loss_per_chars=[],
        gradient_norms=[],
        validation_losses=[],
        validation_loss_per_chars=[],
    )
    adam_state = initialize_adam_state(parameters) if optimizer == "adam" else None
    timestep = 0
    total_steps = epochs * len(windows)
    started_at = time.perf_counter()
    current_learning_rate = learning_rate

    if progress_every is not None:
        print(
            "training_start "
            f"epochs={epochs} windows_per_epoch={len(windows)} total_steps={total_steps} "
            f"vocab_size={vocab_size} hidden_size={hidden_size} sequence_length={sequence_length} "
            f"optimizer={optimizer} learning_rate={learning_rate} "
            f"learning_rate_decay={learning_rate_decay} clip_value={clip_value} stateful={stateful}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        epoch_loss_start = len(history.losses)
        hidden_state = np.zeros((parameters["Waa"].shape[0], 1))

        for window_index, (inputs, targets) in enumerate(windows, start=1):
            timestep += 1
            loss, gradient_norm, hidden_state, adam_state = train_step(
                parameters,
                inputs,
                targets,
                learning_rate=current_learning_rate,
                optimizer=optimizer,
                adam_state=adam_state,
                adam_timestep=timestep,
                clip_value=clip_value,
                initial_hidden_state=hidden_state if stateful else None,
            )
            history.losses.append(loss)
            history.loss_per_chars.append(loss / sequence_length)
            history.gradient_norms.append(gradient_norm)

            if progress_every is not None and (
                timestep == 1 or timestep % progress_every == 0 or timestep == total_steps
            ):
                elapsed = time.perf_counter() - started_at
                steps_per_second = timestep / elapsed if elapsed > 0 else 0.0
                print(
                    "training_progress "
                    f"epoch={epoch}/{epochs} window={window_index}/{len(windows)} "
                    f"step={timestep}/{total_steps} loss={loss:.4f} "
                    f"loss_per_char={loss / sequence_length:.4f} "
                    f"grad_norm={gradient_norm:.4f} elapsed_sec={elapsed:.1f} "
                    f"steps_per_sec={steps_per_second:.2f} lr={current_learning_rate:.6f}",
                    flush=True,
                )

        validation_loss = None
        if validation_data is not None and epoch % validation_every_epochs == 0:
            validation_loss = evaluate_loss(parameters, validation_data, sequence_length)
            history.validation_losses.append(validation_loss)
            history.validation_loss_per_chars.append(validation_loss / sequence_length)

        if progress_every is not None:
            epoch_losses = history.losses[epoch_loss_start:]
            epoch_grad_norms = history.gradient_norms[epoch_loss_start:]
            message = (
                "epoch_complete "
                f"epoch={epoch}/{epochs} avg_loss={float(np.mean(epoch_losses)):.4f} "
                f"avg_loss_per_char={float(np.mean(epoch_losses)) / sequence_length:.4f} "
                f"last_loss={epoch_losses[-1]:.4f} "
                f"avg_grad_norm={float(np.mean(epoch_grad_norms)):.4f} "
                f"epoch_sec={time.perf_counter() - epoch_started_at:.1f} "
                f"lr={current_learning_rate:.6f}"
            )
            if validation_loss is not None:
                message += (
                    f" val_loss={validation_loss:.4f} "
                    f"val_loss_per_char={validation_loss / sequence_length:.4f}"
                )
            print(message, flush=True)

        current_learning_rate *= learning_rate_decay

    if checkpoint_path is not None:
        save_parameters(checkpoint_path, parameters)
        if progress_every is not None:
            print(f"checkpoint_saved path={checkpoint_path}", flush=True)

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
