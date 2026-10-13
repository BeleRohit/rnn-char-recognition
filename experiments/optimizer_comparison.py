"""Experiment B: compare SGD and Adam."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import build_vocabulary, encode_text, load_text
from src.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=30)
    args = parser.parse_args()

    text = load_text(args.data)
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)

    for optimizer, learning_rate in [("sgd", 0.1), ("adam", 0.01)]:
        _, history = train(
            encoded,
            vocab_size=len(chars),
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=0,
        )
        print(
            f"optimizer={optimizer} "
            f"initial_loss={history.losses[0]:.4f} "
            f"final_loss={history.losses[-1]:.4f} "
            f"last_grad_norm={history.gradient_norms[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
