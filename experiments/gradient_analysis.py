"""Experiment C: inspect gradient behavior with and without clipping."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.logging import append_experiment_results
from src.preprocessing import build_vocabulary, encode_text, load_text
from src.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=100)
    args = parser.parse_args()

    text = load_text(args.data)
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)
    rows = []

    for label, clip_value in [("clipped", 5.0), ("unclipped", None)]:
        start = time.perf_counter()
        _, history = train(
            encoded,
            vocab_size=len(chars),
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=0.01,
            optimizer="adam",
            clip_value=clip_value,
            seed=0,
        )
        elapsed = time.perf_counter() - start
        row = {
            "run": label,
            "clip_value": clip_value,
            "max_grad_norm": f"{max(history.gradient_norms):.4f}",
            "final_grad_norm": f"{history.gradient_norms[-1]:.4f}",
            "final_loss": f"{history.losses[-1]:.4f}",
            "final_loss_per_char": f"{history.loss_per_chars[-1]:.4f}",
            "seconds": f"{elapsed:.2f}",
        }
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    append_experiment_results(
        "Experiment C: Gradient Behavior",
        metadata={
            "data": args.data,
            "characters": len(text),
            "vocab_size": len(chars),
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "sequence_length": args.sequence_length,
            "optimizer": "adam",
            "learning_rate": 0.01,
        },
        rows=rows,
    )
    print("results_appended path=experiments/RESULTS.md")


if __name__ == "__main__":
    main()
