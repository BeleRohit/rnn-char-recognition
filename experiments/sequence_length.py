"""Experiment A: compare sequence lengths."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.preprocessing import build_vocabulary, encode_text, load_text
from src.training import evaluate_loss, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    text = load_text(args.data)
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)

    for sequence_length in [10, 30, 100]:
        start = time.perf_counter()
        parameters, history = train(
            encoded,
            vocab_size=len(chars),
            hidden_size=args.hidden_size,
            sequence_length=sequence_length,
            epochs=args.epochs,
            learning_rate=0.01,
            optimizer="adam",
            seed=0,
        )
        elapsed = time.perf_counter() - start
        eval_loss = evaluate_loss(parameters, encoded, sequence_length)
        print(
            f"sequence_length={sequence_length} "
            f"train_loss={history.losses[-1]:.4f} "
            f"eval_loss={eval_loss:.4f} "
            f"last_grad_norm={history.gradient_norms[-1]:.4f} "
            f"seconds={elapsed:.2f}"
        )


if __name__ == "__main__":
    main()
