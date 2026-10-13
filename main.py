"""Command-line entry point for small character-RNN runs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.preprocessing import build_vocabulary, encode_text, load_text
from src.sampling import generate_text
from src.training import train
from src.utils import load_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy character-level RNN.")
    parser.add_argument("--data", type=Path, required=True, help="Path to a UTF-8 text file.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--learning-rate-decay",
        type=float,
        default=1.0,
        help="Multiply the learning rate by this value after each epoch.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of text held out for validation during training.",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Evaluate validation loss every N epochs.",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Reset hidden state to zeros for every training window.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.npz"))
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Load model parameters from this .npz checkpoint.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Skip training and generate text from --load-checkpoint.",
    )
    parser.add_argument("--sample-start", type=str, default=None)
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print one progress line every N training steps. Use 0 to disable.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    print(f"loading_data path={args.data}", flush=True)
    text = load_text(args.data)
    chars, char_to_idx, idx_to_char = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)
    progress_every = None if args.progress_every == 0 else args.progress_every
    if not 0.0 <= args.validation_split < 1.0:
        parser.error("--validation-split must be in [0.0, 1.0).")
    split_index = int(len(encoded) * (1.0 - args.validation_split))
    train_encoded = encoded[:split_index]
    validation_encoded = encoded[split_index:] if args.validation_split > 0 else None

    print(
        f"data_loaded characters={len(text)} vocab_size={len(chars)} "
        f"train_characters={len(train_encoded)} "
        f"validation_characters={0 if validation_encoded is None else len(validation_encoded)} "
        f"unique_chars={repr(''.join(chars))}",
        flush=True,
    )

    if args.sample_only:
        if args.load_checkpoint is None:
            parser.error("--sample-only requires --load-checkpoint.")
        if args.sample_start is None:
            parser.error("--sample-only requires --sample-start.")
        print(f"loading_checkpoint path={args.load_checkpoint}", flush=True)
        parameters = load_parameters(args.load_checkpoint)
        print("checkpoint_loaded", flush=True)
    else:
        if args.load_checkpoint is not None:
            print(f"loading_checkpoint path={args.load_checkpoint}", flush=True)
            parameters = load_parameters(args.load_checkpoint)
            if parameters["Wax"].shape[1] != len(chars):
                raise ValueError(
                    "Checkpoint vocabulary size does not match the loaded data. "
                    "Use the same text data that was used for training."
                )
            print("checkpoint_loaded continuing_training=true", flush=True)
        else:
            parameters = None

        parameters, history = train(
            train_encoded,
            vocab_size=len(chars),
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            progress_every=progress_every,
            initial_parameters=parameters,
            validation_data=validation_encoded,
            validation_every_epochs=args.validation_every_epochs,
            stateful=not args.stateless,
            learning_rate_decay=args.learning_rate_decay,
        )

        validation_summary = ""
        if history.validation_losses:
            validation_summary = (
                f" final_val_loss={history.validation_losses[-1]:.4f} "
                f"final_val_loss_per_char={history.validation_loss_per_chars[-1]:.4f}"
            )
        print(
            "training_complete "
            f"vocab_size={len(chars)} steps={len(history.losses)} "
            f"initial_loss={history.losses[0]:.4f} final_loss={history.losses[-1]:.4f} "
            f"initial_loss_per_char={history.loss_per_chars[0]:.4f} "
            f"final_loss_per_char={history.loss_per_chars[-1]:.4f}"
            f"{validation_summary} "
            f"total_sec={time.perf_counter() - started_at:.1f} checkpoint={args.checkpoint}",
            flush=True,
        )

    if args.sample_start is not None:
        print(
            f"sampling_start start_char={args.sample_start!r} length={args.sample_length} "
            f"temperature={args.temperature}",
            flush=True,
        )
        sample = generate_text(
            parameters,
            start_char=args.sample_start,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            num_chars=args.sample_length,
            temperature=args.temperature,
            seed=args.seed,
        )
        print("sample_output", flush=True)
        print(sample)


if __name__ == "__main__":
    main()
