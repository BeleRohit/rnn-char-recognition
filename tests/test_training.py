import numpy as np

from src.preprocessing import build_vocabulary, encode_text
from src.training import train, train_step
from src.utils import load_parameters, save_parameters


def test_train_step_reduces_loss_on_repeated_example():
    text = "abababababab"
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)
    inputs = encoded[:4]
    targets = encoded[1:5]

    params, _ = train(
        encoded,
        vocab_size=len(chars),
        hidden_size=8,
        sequence_length=4,
        epochs=1,
        learning_rate=0.05,
        optimizer="adam",
        seed=0,
    )
    initial_loss, _, hidden_state, adam_state = train_step(
        params,
        inputs,
        targets,
        learning_rate=0.05,
        optimizer="adam",
        adam_timestep=1,
    )
    final_loss = initial_loss
    for step in range(2, 80):
        final_loss, _, hidden_state, adam_state = train_step(
            params,
            inputs,
            targets,
            learning_rate=0.05,
            optimizer="adam",
            adam_state=adam_state,
            adam_timestep=step,
            initial_hidden_state=hidden_state,
        )

    assert final_loss < initial_loss * 0.2


def test_training_history_records_loss_and_gradient_norms():
    text = "hello hello hello"
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)

    _, history = train(
        encoded,
        vocab_size=len(chars),
        hidden_size=6,
        sequence_length=5,
        epochs=2,
        learning_rate=0.01,
        optimizer="sgd",
        seed=1,
    )

    assert len(history.losses) == len(history.gradient_norms)
    assert len(history.losses) == len(history.loss_per_chars)
    assert len(history.losses) > 0
    assert np.all(np.isfinite(history.losses))
    assert np.all(np.isfinite(history.loss_per_chars))
    assert np.all(np.isfinite(history.gradient_norms))


def test_training_records_validation_loss_when_validation_data_is_provided():
    text = "hello hello hello hello"
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)

    _, history = train(
        encoded[:14],
        vocab_size=len(chars),
        hidden_size=6,
        sequence_length=4,
        epochs=2,
        learning_rate=0.01,
        optimizer="adam",
        seed=1,
        validation_data=encoded[10:],
    )

    assert len(history.validation_losses) == 2
    assert len(history.validation_loss_per_chars) == 2
    assert np.all(np.isfinite(history.validation_losses))


def test_checkpoint_restoration_round_trip(tmp_path):
    text = "abcabcabc"
    chars, char_to_idx, _ = build_vocabulary(text)
    encoded = encode_text(text, char_to_idx)
    checkpoint = tmp_path / "model.npz"

    params, _ = train(
        encoded,
        vocab_size=len(chars),
        hidden_size=5,
        sequence_length=3,
        epochs=1,
        learning_rate=0.01,
        optimizer="adam",
        seed=2,
        checkpoint_path=checkpoint,
    )
    loaded = load_parameters(checkpoint)

    assert set(loaded) == set(params)
    for name in params:
        np.testing.assert_allclose(loaded[name], params[name])


def test_save_and_load_parameters_round_trip(tmp_path):
    params = {
        "Wax": np.array([[1.0, 2.0]]),
        "Waa": np.array([[3.0]]),
        "Wya": np.array([[4.0]]),
        "ba": np.array([[5.0]]),
        "by": np.array([[6.0]]),
    }
    path = tmp_path / "manual.npz"

    save_parameters(path, params)
    loaded = load_parameters(path)

    for name in params:
        np.testing.assert_allclose(loaded[name], params[name])
