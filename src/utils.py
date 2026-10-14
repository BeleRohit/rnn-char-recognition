"""General project utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.rnn import ParameterDict


def save_parameters(path: str | Path, parameters: ParameterDict) -> None:
    """Save model parameters to a compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **parameters)


def load_parameters(path: str | Path) -> ParameterDict:
    """Load model parameters saved by save_parameters."""
    with np.load(Path(path)) as data:
        return {name: data[name] for name in data.files if not name.startswith("__")}


def save_checkpoint(path: str | Path, parameters: ParameterDict, metadata: dict[str, object]) -> None:
    """Save model parameters and JSON metadata to a compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(output_path, __metadata_json__=np.array(metadata_json), **parameters)


def load_checkpoint(path: str | Path) -> tuple[ParameterDict, dict[str, object]]:
    """Load parameters and metadata from a checkpoint."""
    with np.load(Path(path), allow_pickle=False) as data:
        parameters = {name: data[name] for name in data.files if not name.startswith("__")}
        metadata_json = str(data["__metadata_json__"]) if "__metadata_json__" in data.files else "{}"
    return parameters, json.loads(metadata_json)


def iter_sequence_windows(
    encoded_text: np.ndarray | list[int], sequence_length: int, stride: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create many shifted input/target windows from encoded text."""
    encoded_array = np.asarray(encoded_text, dtype=np.int64)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    step = sequence_length if stride is None else stride
    if step <= 0:
        raise ValueError("stride must be positive.")

    windows = []
    for start in range(0, encoded_array.size - sequence_length, step):
        chunk = encoded_array[start : start + sequence_length + 1]
        windows.append((chunk[:-1], chunk[1:]))
    return windows


def plot_training_curves(
    losses: list[float],
    gradient_norms: list[float] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Plot loss and optional gradient norm curves."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    axis.plot(losses, label="loss")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")

    if gradient_norms is not None:
        twin = axis.twinx()
        twin.plot(gradient_norms, color="tab:orange", label="gradient norm")
        twin.set_ylabel("gradient norm")

    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)
