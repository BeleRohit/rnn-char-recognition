"""Forward and backward propagation for a vanilla character-level RNN."""

from __future__ import annotations

import numpy as np


ParameterDict = dict[str, np.ndarray]


def initialize_parameters(
    n_x: int, n_a: int, n_y: int | None = None, seed: int | None = None
) -> ParameterDict:
    """Initialize vanilla RNN parameters.

    Dimensions:
        Wax: (n_a, n_x)
        Waa: (n_a, n_a)
        Wya: (n_y, n_a)
        ba:  (n_a, 1)
        by:  (n_y, 1)
    """
    if n_x <= 0 or n_a <= 0:
        raise ValueError("n_x and n_a must be positive.")

    output_size = n_x if n_y is None else n_y
    if output_size <= 0:
        raise ValueError("n_y must be positive.")

    rng = np.random.default_rng(seed)
    scale = 0.01
    return {
        "Wax": rng.normal(0.0, scale, size=(n_a, n_x)),
        "Waa": rng.normal(0.0, scale, size=(n_a, n_a)),
        "Wya": rng.normal(0.0, scale, size=(output_size, n_a)),
        "ba": np.zeros((n_a, 1)),
        "by": np.zeros((output_size, 1)),
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities for a column vector or matrix."""
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def rnn_cell_forward(
    x_t: np.ndarray, a_prev: np.ndarray, parameters: ParameterDict
) -> tuple[np.ndarray, np.ndarray]:
    """Run one RNN time step.

    Args:
        x_t: Input one-hot column vector with shape (n_x, 1).
        a_prev: Previous hidden state with shape (n_a, 1).
        parameters: RNN parameters.

    Returns:
        a_next: Current hidden state with shape (n_a, 1).
        y_hat_t: Output probabilities with shape (n_y, 1).
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(Wax @ x_t + Waa @ a_prev + ba)
    logits = Wya @ a_next + by
    y_hat_t = stable_softmax(logits)
    return a_next, y_hat_t


def rnn_forward(
    X: np.ndarray, a0: np.ndarray, parameters: ParameterDict
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Run forward propagation over a full sequence.

    Args:
        X: One-hot inputs with shape (n_x, T).
        a0: Initial hidden state with shape (n_a, 1).
        parameters: RNN parameters.

    Returns:
        hidden_states: Dict containing a<-1> at key -1 and a<t> for t in [0, T - 1].
        predictions: Dict containing y_hat<t> for each time step t.
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (n_x, T).")
    if a0.ndim != 2 or a0.shape[1] != 1:
        raise ValueError("a0 must have shape (n_a, 1).")
    if X.shape[0] != parameters["Wax"].shape[1]:
        raise ValueError("X first dimension must match n_x.")
    if a0.shape[0] != parameters["Waa"].shape[0]:
        raise ValueError("a0 first dimension must match n_a.")

    hidden_states = {-1: a0}
    predictions: dict[int, np.ndarray] = {}

    a_prev = a0
    for t in range(X.shape[1]):
        x_t = X[:, [t]]
        a_next, y_hat_t = rnn_cell_forward(x_t, a_prev, parameters)
        hidden_states[t] = a_next
        predictions[t] = y_hat_t
        a_prev = a_next

    return hidden_states, predictions


def bptt(
    X: np.ndarray,
    targets: np.ndarray | list[int],
    parameters: ParameterDict,
    hidden_states: dict[int, np.ndarray],
    predictions: dict[int, np.ndarray],
) -> ParameterDict:
    """Backpropagate sequence loss through time.

    The same weights are reused at every time step. For that reason, each
    parameter gradient is the sum of its contribution across all positions in
    the sequence.
    """
    target_array = np.asarray(targets, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_x, T).")
    if target_array.ndim != 1:
        raise ValueError("targets must be a one-dimensional sequence.")
    if X.shape[1] != target_array.size:
        raise ValueError("X and targets must have the same sequence length.")

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]

    gradients = {
        "dWax": np.zeros_like(Wax),
        "dWaa": np.zeros_like(Waa),
        "dWya": np.zeros_like(Wya),
        "dba": np.zeros_like(parameters["ba"]),
        "dby": np.zeros_like(parameters["by"]),
    }

    da_next = np.zeros_like(hidden_states[0])

    for t in reversed(range(X.shape[1])):
        dy = predictions[t].copy()
        dy[int(target_array[t]), 0] -= 1.0

        gradients["dWya"] += dy @ hidden_states[t].T
        gradients["dby"] += dy

        da = Wya.T @ dy + da_next
        dz = (1.0 - hidden_states[t] ** 2) * da

        x_t = X[:, [t]]
        a_prev = hidden_states[t - 1]
        gradients["dWax"] += dz @ x_t.T
        gradients["dWaa"] += dz @ a_prev.T
        gradients["dba"] += dz

        da_next = Waa.T @ dz

    return gradients


def clip_gradients(gradients: ParameterDict, max_value: float) -> ParameterDict:
    """Clip each gradient element into [-max_value, max_value]."""
    if max_value <= 0:
        raise ValueError("max_value must be positive.")

    return {
        name: np.clip(gradient, -max_value, max_value)
        for name, gradient in gradients.items()
    }


def global_gradient_norm(gradients: ParameterDict) -> float:
    """Compute the L2 norm over every gradient tensor."""
    squared_sum = sum(float(np.sum(gradient * gradient)) for gradient in gradients.values())
    return float(np.sqrt(squared_sum))


def numerical_gradient_check(
    X: np.ndarray,
    targets: np.ndarray | list[int],
    parameters: ParameterDict,
    epsilon: float = 1e-7,
) -> float:
    """Compare BPTT gradients with finite-difference gradients.

    Returns the relative difference ||g - g_num|| / (||g|| + ||g_num||).
    Use this with very small deterministic networks because it is slow.
    """
    from src.losses import sequence_cross_entropy_loss

    a0 = np.zeros((parameters["Waa"].shape[0], 1))
    hidden_states, predictions = rnn_forward(X, a0, parameters)
    analytical = bptt(X, targets, parameters, hidden_states, predictions)

    analytical_vector = []
    numerical_vector = []
    gradient_names = {
        "Wax": "dWax",
        "Waa": "dWaa",
        "Wya": "dWya",
        "ba": "dba",
        "by": "dby",
    }

    for param_name, grad_name in gradient_names.items():
        parameter = parameters[param_name]
        for index in np.ndindex(parameter.shape):
            original_value = parameter[index]

            parameter[index] = original_value + epsilon
            _, plus_predictions = rnn_forward(X, a0, parameters)
            plus_loss = sequence_cross_entropy_loss(plus_predictions, targets)

            parameter[index] = original_value - epsilon
            _, minus_predictions = rnn_forward(X, a0, parameters)
            minus_loss = sequence_cross_entropy_loss(minus_predictions, targets)

            parameter[index] = original_value

            numerical = (plus_loss - minus_loss) / (2.0 * epsilon)
            analytical_vector.append(float(analytical[grad_name][index]))
            numerical_vector.append(float(numerical))

    analytical_array = np.array(analytical_vector)
    numerical_array = np.array(numerical_vector)
    numerator = np.linalg.norm(analytical_array - numerical_array)
    denominator = np.linalg.norm(analytical_array) + np.linalg.norm(numerical_array)
    return float(numerator / denominator)
