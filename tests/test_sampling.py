import numpy as np

from src.preprocessing import build_vocabulary
from src.rnn import initialize_parameters
from src.sampling import apply_temperature, generate_text, sample_indices


def test_apply_temperature_returns_probability_distribution():
    probabilities = np.array([[0.2], [0.3], [0.5]])

    scaled = apply_temperature(probabilities, temperature=0.7)

    assert scaled.shape == probabilities.shape
    assert np.all(scaled > 0)
    np.testing.assert_allclose(np.sum(scaled), 1.0)


def test_sample_indices_is_deterministic_with_seed():
    params = initialize_parameters(n_x=3, n_a=4, n_y=3, seed=0)

    first = sample_indices(params, start_idx=1, num_chars=8, seed=123)
    second = sample_indices(params, start_idx=1, num_chars=8, seed=123)

    assert first == second
    assert len(first) == 9


def test_generate_text_returns_start_plus_requested_characters():
    chars, char_to_idx, idx_to_char = build_vocabulary("abc")
    params = initialize_parameters(n_x=len(chars), n_a=4, n_y=len(chars), seed=0)

    text = generate_text(params, "a", char_to_idx, idx_to_char, num_chars=5, seed=7)

    assert text[0] == "a"
    assert len(text) == 6
    assert set(text).issubset(set(chars))
