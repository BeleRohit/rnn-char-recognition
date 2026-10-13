import numpy as np

from src.preprocessing import one_hot_encode
from src.rnn import initialize_parameters, rnn_cell_forward, rnn_forward, stable_softmax


def test_initialize_parameters_dimensions():
    params = initialize_parameters(n_x=5, n_a=7, n_y=5, seed=123)

    assert params["Wax"].shape == (7, 5)
    assert params["Waa"].shape == (7, 7)
    assert params["Wya"].shape == (5, 7)
    assert params["ba"].shape == (7, 1)
    assert params["by"].shape == (5, 1)


def test_stable_softmax_handles_large_logits():
    logits = np.array([[1000.0], [1001.0], [1002.0]])

    probs = stable_softmax(logits)

    assert probs.shape == (3, 1)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(np.sum(probs, axis=0), np.array([1.0]))
    assert probs[2, 0] > probs[1, 0] > probs[0, 0]


def test_rnn_cell_forward_matches_manual_computation():
    params = {
        "Wax": np.array([[0.1, -0.2], [0.3, 0.4]]),
        "Waa": np.array([[0.2, 0.0], [-0.1, 0.1]]),
        "Wya": np.array([[0.5, -0.3], [0.1, 0.2]]),
        "ba": np.array([[0.01], [-0.02]]),
        "by": np.array([[0.03], [-0.01]]),
    }
    x_t = np.array([[1.0], [0.0]])
    a_prev = np.array([[0.2], [-0.1]])

    a_next, y_hat = rnn_cell_forward(x_t, a_prev, params)

    expected_a = np.tanh(params["Wax"] @ x_t + params["Waa"] @ a_prev + params["ba"])
    expected_y = stable_softmax(params["Wya"] @ expected_a + params["by"])
    np.testing.assert_allclose(a_next, expected_a)
    np.testing.assert_allclose(y_hat, expected_y)


def test_rnn_forward_returns_state_and_prediction_for_each_time_step():
    X = one_hot_encode([0, 1, 2], vocab_size=3)
    params = initialize_parameters(n_x=3, n_a=4, n_y=3, seed=42)
    a0 = np.zeros((4, 1))

    hidden_states, predictions = rnn_forward(X, a0, params)

    assert set(hidden_states.keys()) == {-1, 0, 1, 2}
    assert set(predictions.keys()) == {0, 1, 2}
    assert hidden_states[-1].shape == (4, 1)
    for t in range(3):
        assert hidden_states[t].shape == (4, 1)
        assert predictions[t].shape == (3, 1)
        np.testing.assert_allclose(np.sum(predictions[t], axis=0), np.array([1.0]))
