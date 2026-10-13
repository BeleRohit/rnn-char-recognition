"""Optimization algorithms for RNN parameter updates."""

from __future__ import annotations

import numpy as np

from src.rnn import ParameterDict


GRADIENT_TO_PARAMETER = {
    "dWax": "Wax",
    "dWaa": "Waa",
    "dWya": "Wya",
    "dba": "ba",
    "dby": "by",
}


def sgd_update(parameters: ParameterDict, gradients: ParameterDict, learning_rate: float) -> ParameterDict:
    """Apply one in-place stochastic gradient descent update."""
    for gradient_name, parameter_name in GRADIENT_TO_PARAMETER.items():
        parameters[parameter_name] -= learning_rate * gradients[gradient_name]
    return parameters


def initialize_adam_state(parameters: ParameterDict) -> dict[str, dict[str, np.ndarray]]:
    """Create Adam first- and second-moment buffers."""
    return {
        "v": {name: np.zeros_like(value) for name, value in parameters.items()},
        "s": {name: np.zeros_like(value) for name, value in parameters.items()},
    }


def adam_update(
    parameters: ParameterDict,
    gradients: ParameterDict,
    state: dict[str, dict[str, np.ndarray]],
    timestep: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> ParameterDict:
    """Apply one in-place Adam update."""
    if timestep <= 0:
        raise ValueError("timestep must be positive for Adam bias correction.")

    for gradient_name, parameter_name in GRADIENT_TO_PARAMETER.items():
        gradient = gradients[gradient_name]
        state["v"][parameter_name] = beta1 * state["v"][parameter_name] + (1.0 - beta1) * gradient
        state["s"][parameter_name] = beta2 * state["s"][parameter_name] + (1.0 - beta2) * (gradient * gradient)

        v_corrected = state["v"][parameter_name] / (1.0 - beta1**timestep)
        s_corrected = state["s"][parameter_name] / (1.0 - beta2**timestep)
        parameters[parameter_name] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return parameters
