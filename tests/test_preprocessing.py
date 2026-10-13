import numpy as np
import pytest

from src.preprocessing import (
    build_vocabulary,
    decode_indices,
    encode_text,
    make_input_target_sequences,
    one_hot_encode,
)


def test_build_vocabulary_is_deterministic():
    chars, char_to_idx, idx_to_char = build_vocabulary("banana")

    assert chars == ["a", "b", "n"]
    assert char_to_idx == {"a": 0, "b": 1, "n": 2}
    assert idx_to_char == {0: "a", 1: "b", 2: "n"}


def test_encode_decode_round_trip():
    text = "abba"
    _, char_to_idx, idx_to_char = build_vocabulary(text)

    encoded = encode_text(text, char_to_idx)
    decoded = decode_indices(encoded, idx_to_char)

    np.testing.assert_array_equal(encoded, np.array([0, 1, 1, 0]))
    assert decoded == text


def test_one_hot_encode_columns_are_character_vectors():
    one_hot = one_hot_encode([1, 0, 2], vocab_size=3)

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert one_hot.shape == (3, 3)
    np.testing.assert_array_equal(one_hot, expected)


def test_make_input_target_sequences_shift_by_one_character():
    inputs, targets = make_input_target_sequences([0, 1, 2, 3, 4], sequence_length=4)

    np.testing.assert_array_equal(inputs, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(targets, np.array([1, 2, 3, 4]))


def test_empty_vocabulary_raises_error():
    with pytest.raises(ValueError, match="empty text"):
        build_vocabulary("")
