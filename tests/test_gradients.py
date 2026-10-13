import numpy as np

from src.losses import cross_entropy_loss, sequence_cross_entropy_loss
from src.optimizers import adam_update, initialize_adam_state, sgd_update
from src.preprocessing import one_hot_encode
from src.rnn import (
    bptt,
    clip_gradients,
    global_gradient_norm,
    initialize_parameters,
    numerical_gradient_check,
    rnn_forward,
)


def test_cross_entropy_loss_is_finite():
    y_hat = np.array([[0.1], [0.8], [0.1]])

    loss = cross_entropy_loss(y_hat, target_idx=1)

    assert np.isfinite(loss)
    np.testing.assert_allclose(loss, -np.log(0.8 + 1e-12))


def test_bptt_gradient_shapes_match_parameters():
    params = initialize_parameters(n_x=3, n_a=4, n_y=3, seed=1)
    X = one_hot_encode([0, 1, 2], vocab_size=3)
    targets = np.array([1, 2, 0])
    hidden_states, predictions = rnn_forward(X, np.zeros((4, 1)), params)

    gradients = bptt(X, targets, params, hidden_states, predictions)

    assert gradients["dWax"].shape == params["Wax"].shape
    assert gradients["dWaa"].shape == params["Waa"].shape
    assert gradients["dWya"].shape == params["Wya"].shape
    assert gradients["dba"].shape == params["ba"].shape
    assert gradients["dby"].shape == params["by"].shape


def test_numerical_gradient_check_passes_for_tiny_network():
    params = initialize_parameters(n_x=2, n_a=2, n_y=2, seed=3)
    X = one_hot_encode([0, 1], vocab_size=2)
    targets = np.array([1, 0])

    difference = numerical_gradient_check(X, targets, params)

    assert difference < 1e-7


def test_gradient_clipping_limits_every_element():
    gradients = {
        "dWax": np.array([[10.0, -10.0], [0.5, -0.5]]),
        "dWaa": np.array([[7.0]]),
    }

    clipped = clip_gradients(gradients, max_value=1.0)

    assert np.max(clipped["dWax"]) <= 1.0
    assert np.min(clipped["dWax"]) >= -1.0
    assert clipped["dWaa"][0, 0] == 1.0


def test_global_gradient_norm_combines_all_arrays():
    gradients = {"a": np.array([[3.0]]), "b": np.array([[4.0]])}

    assert global_gradient_norm(gradients) == 5.0


def test_sgd_update_changes_parameters_in_gradient_direction():
    params = initialize_parameters(n_x=2, n_a=2, n_y=2, seed=0)
    original_wax = params["Wax"].copy()
    gradients = {
        "dWax": np.ones_like(params["Wax"]),
        "dWaa": np.ones_like(params["Waa"]),
        "dWya": np.ones_like(params["Wya"]),
        "dba": np.ones_like(params["ba"]),
        "dby": np.ones_like(params["by"]),
    }

    sgd_update(params, gradients, learning_rate=0.1)

    np.testing.assert_allclose(params["Wax"], original_wax - 0.1)


def test_adam_update_changes_parameters_and_state():
    params = initialize_parameters(n_x=2, n_a=2, n_y=2, seed=0)
    state = initialize_adam_state(params)
    original_wya = params["Wya"].copy()
    gradients = {
        "dWax": np.ones_like(params["Wax"]),
        "dWaa": np.ones_like(params["Waa"]),
        "dWya": np.ones_like(params["Wya"]),
        "dba": np.ones_like(params["ba"]),
        "dby": np.ones_like(params["by"]),
    }

    adam_update(params, gradients, state, timestep=1, learning_rate=0.01)

    assert np.any(state["v"]["Wya"] != 0.0)
    assert np.any(state["s"]["Wya"] != 0.0)
    assert not np.allclose(params["Wya"], original_wya)


def test_sequence_cross_entropy_loss_sums_time_steps():
    predictions = {
        0: np.array([[0.25], [0.75]]),
        1: np.array([[0.5], [0.5]]),
    }

    loss = sequence_cross_entropy_loss(predictions, [1, 0])

    np.testing.assert_allclose(loss, -np.log(0.75 + 1e-12) - np.log(0.5 + 1e-12))
