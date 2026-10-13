"""Autoregressive text generation for a trained character RNN."""

from __future__ import annotations

import numpy as np

from src.preprocessing import decode_indices, one_hot_encode
from src.rnn import ParameterDict, rnn_cell_forward, stable_softmax


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability distribution before sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    return stable_softmax(logits)


def sample_indices(
    parameters: ParameterDict,
    start_idx: int,
    num_chars: int,
    temperature: float = 1.0,
    seed: int | None = None,
) -> list[int]:
    """Generate character ids one at a time from an initial character id."""
    if num_chars < 0:
        raise ValueError("num_chars must be non-negative.")

    rng = np.random.default_rng(seed)
    vocab_size = parameters["Wax"].shape[1]
    hidden_size = parameters["Waa"].shape[0]

    current_idx = int(start_idx)
    if current_idx < 0 or current_idx >= vocab_size:
        raise ValueError("start_idx is outside the vocabulary range.")

    generated = [current_idx]
    a_prev = np.zeros((hidden_size, 1))

    for _ in range(num_chars):
        x_t = one_hot_encode([current_idx], vocab_size)
        a_prev, probabilities = rnn_cell_forward(x_t, a_prev, parameters)
        scaled_probabilities = apply_temperature(probabilities, temperature)
        current_idx = int(rng.choice(vocab_size, p=scaled_probabilities.ravel()))
        generated.append(current_idx)

    return generated


def generate_text(
    parameters: ParameterDict,
    start_char: str,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    num_chars: int,
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    """Generate text from a starting character."""
    if start_char not in char_to_idx:
        raise KeyError(f"Character {start_char!r} is not present in the vocabulary.")

    indices = sample_indices(
        parameters,
        start_idx=char_to_idx[start_char],
        num_chars=num_chars,
        temperature=temperature,
        seed=seed,
    )
    return decode_indices(indices, idx_to_char)
