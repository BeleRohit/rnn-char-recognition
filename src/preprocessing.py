"""Text preprocessing utilities for a character-level language model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_text(path: str | Path) -> str:
    """Load raw text from disk."""
    return Path(path).read_text(encoding="utf-8")


def build_vocabulary(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build deterministic character vocabulary mappings from raw text.

    Returns:
        chars: Sorted list of unique characters.
        char_to_idx: Mapping from character to integer index.
        idx_to_char: Mapping from integer index back to character.
    """
    if not text:
        raise ValueError("Cannot build a vocabulary from empty text.")

    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char


def encode_text(text: str, char_to_idx: dict[str, int]) -> np.ndarray:
    """Convert a string into an array of character indices."""
    try:
        return np.array([char_to_idx[char] for char in text], dtype=np.int64)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Character {missing!r} is not present in the vocabulary.") from exc


def decode_indices(indices: np.ndarray | list[int], idx_to_char: dict[int, str]) -> str:
    """Convert character indices back into a string."""
    try:
        return "".join(idx_to_char[int(idx)] for idx in indices)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Index {missing!r} is not present in the vocabulary.") from exc


def one_hot_encode(indices: np.ndarray | list[int], vocab_size: int) -> np.ndarray:
    """Represent indices as columns of one-hot vectors.

    Args:
        indices: Sequence of integer character ids with length T.
        vocab_size: Number of unique characters, n_x.

    Returns:
        Array with shape (vocab_size, T), where each column is one x<t>.
    """
    indices_array = np.asarray(indices, dtype=np.int64)

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if indices_array.ndim != 1:
        raise ValueError("indices must be a one-dimensional sequence.")
    if indices_array.size == 0:
        return np.zeros((vocab_size, 0), dtype=np.float64)
    if indices_array.min() < 0 or indices_array.max() >= vocab_size:
        raise ValueError("indices contain values outside the vocabulary range.")

    one_hot = np.zeros((vocab_size, indices_array.size), dtype=np.float64)
    one_hot[indices_array, np.arange(indices_array.size)] = 1.0
    return one_hot


def make_input_target_sequences(
    encoded_text: np.ndarray | list[int], sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-shifted language-model inputs and targets.

    If encoded text is [c0, c1, c2, c3] and sequence_length is 3, then
    inputs are [c0, c1, c2] and targets are [c1, c2, c3].
    """
    encoded_array = np.asarray(encoded_text, dtype=np.int64)

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if encoded_array.ndim != 1:
        raise ValueError("encoded_text must be a one-dimensional sequence.")
    if encoded_array.size < sequence_length + 1:
        raise ValueError("encoded_text must contain at least sequence_length + 1 values.")

    inputs = encoded_array[:sequence_length]
    targets = encoded_array[1 : sequence_length + 1]
    return inputs, targets
