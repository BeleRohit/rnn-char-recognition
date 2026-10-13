"""Command-line entry point for small character-RNN runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import build_vocabulary, encode_text, load_text
from src.sampling import generate_text
from src.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy character-level RNN.")
    parser.add_argument("--data", type=Path, required=True, help="Path to a UTF-8 text file.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.npz"))
    parser.add_argument("--sample-start", type=str, default=None)
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    text = load_text(args.data)
    chars, char_to_idx, idx_to_char = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)

    parameters, history = train(
        encoded,
        vocab_size=len(chars),
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )

    print(f"vocab_size={len(chars)}")
    print(f"steps={len(history.losses)}")
    print(f"initial_loss={history.losses[0]:.4f}")
    print(f"final_loss={history.losses[-1]:.4f}")
    print(f"checkpoint={args.checkpoint}")

    if args.sample_start is not None:
        sample = generate_text(
            parameters,
            start_char=args.sample_start,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            num_chars=args.sample_length,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(sample)


if __name__ == "__main__":
    main()
