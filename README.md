# Character-Level Shakespeare RNN From Scratch

A from-scratch character-level language model implemented with NumPy. The goal
is to understand recurrent neural networks by building the full stack manually:
preprocessing, forward propagation, BPTT, gradient clipping, Adam, checkpointing,
validation tracking, temperature sampling, and controlled experiments.

No PyTorch. No TensorFlow. No autograd. Just NumPy and the math.

## What It Does

The model learns to predict the next character in raw text. Given a sequence
like:

```text
To be
```

it learns next-character probabilities one step at a time:

```text
T -> o
o -> " "
" " -> b
b -> e
```

After training, it can generate new Shakespeare-like text autoregressively.

## Current Results

Measured on the Tiny Shakespeare dataset with a vanilla RNN:

| Run | Hidden Size | Seq Len | Train Loss/Char | Val Loss/Char | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| h128 seq30 | 128 | 30 | 1.7915 | 2.1922 | 14.5 min |
| h128 seq50 continuation | 128 | 50 | 1.7158 | 2.1348 | 88.3 sec |

Random guessing over the 65-character vocabulary starts near:

```text
log(65) = 4.1744 loss/char
```

So the model has learned meaningful character-level structure, though it is
still a small vanilla RNN and does not produce consistently coherent language.

## Example Output

Example from the `h128` checkpoint at temperature `0.8`:

```text
THAASIA:
Being what it inture not such. Astonster to men that think gove do againce
thou on hide ued sher well you will.
```

This is not semantically polished, but it shows learned dialogue formatting,
speaker-name structure, punctuation rhythm, and Shakespeare-like word shapes.

## Architecture

The vanilla RNN equations are:

```text
a<t> = tanh(Wax x<t> + Waa a<t-1> + ba)
z<t> = Wya a<t> + by
y_hat<t> = softmax(z<t>)
```

Shape reference:

| Symbol | Meaning | Shape |
| --- | --- | --- |
| `x<t>` | one-hot input character | `(n_x, 1)` |
| `a<t>` | hidden state | `(n_a, 1)` |
| `y_hat<t>` | next-character probabilities | `(n_y, 1)` |
| `Wax` | input-to-hidden weights | `(n_a, n_x)` |
| `Waa` | hidden-to-hidden weights | `(n_a, n_a)` |
| `Wya` | hidden-to-output weights | `(n_y, n_a)` |

Recurrence sketch:

```text
            previous memory
                 |
                 v
x<t> -> Wax -> tanh -> a<t> -> Wya -> softmax -> y_hat<t>
                 ^
                 |
                Waa
```

## Implemented

- Raw text loading and deterministic character vocabulary
- Integer encoding and one-hot encoding
- Vanilla RNN forward propagation
- Numerically stable softmax
- Sequence cross-entropy loss
- Backpropagation Through Time
- Gradient accumulation across shared recurrent weights
- Gradient clipping
- Numerical gradient checking
- SGD and Adam optimizers
- Stateful truncated BPTT across adjacent windows
- Validation split and validation loss tracking
- Learning-rate decay
- Checkpoints with metadata
- Temperature-controlled sampling
- Training curve export
- Experiment logging to `experiments/RESULTS.md`

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Get the Tiny Shakespeare dataset:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/shakespeare.txt
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Current verification:

```text
26 passed
```

## Train

Small smoke run:

```bash
make train-small
```

Stronger run:

```bash
.venv/bin/python main.py \
  --data data/shakespeare.txt \
  --epochs 10 \
  --hidden-size 128 \
  --sequence-length 50 \
  --learning-rate 0.005 \
  --learning-rate-decay 0.95 \
  --optimizer adam \
  --validation-split 0.1 \
  --checkpoint checkpoints/shakespeare-rnn-h128.npz \
  --plot-path plots/h128-loss.png \
  --record-results \
  --progress-every 1000
```

Continue from a checkpoint:

```bash
.venv/bin/python main.py \
  --data data/shakespeare.txt \
  --load-checkpoint checkpoints/shakespeare-rnn-h128.npz \
  --epochs 5 \
  --hidden-size 128 \
  --sequence-length 50 \
  --learning-rate 0.001 \
  --learning-rate-decay 0.95 \
  --optimizer adam \
  --checkpoint checkpoints/shakespeare-rnn-h128-v2.npz
```

## Sample

For metadata-rich checkpoints created by the current code, sampling does not
need the original data file:

```bash
.venv/bin/python main.py \
  --load-checkpoint checkpoints/shakespeare-rnn-h128.npz \
  --sample-only \
  --sample-start T \
  --sample-length 1000 \
  --temperature 0.7 \
  --seed 42
```

For older checkpoints without metadata, include `--data`:

```bash
.venv/bin/python main.py \
  --data data/shakespeare.txt \
  --load-checkpoint checkpoints/shakespeare-rnn-h128.npz \
  --sample-only \
  --sample-start T \
  --sample-length 1000 \
  --temperature 0.7
```

Temperature guide:

- `0.6`: safer and more repetitive
- `0.7` to `0.8`: usually the best balance
- `1.0`: more varied and noisier

## Experiments

Experiment scripts print results and append Markdown tables to
`experiments/RESULTS.md`.

```bash
.venv/bin/python experiments/optimizer_comparison.py --data data/shakespeare.txt
.venv/bin/python experiments/sequence_length.py --data data/shakespeare.txt
.venv/bin/python experiments/gradient_analysis.py --data data/shakespeare.txt
```

Or run all:

```bash
make experiments
```

## Project Structure

```text
src/
  preprocessing.py   text loading, vocabulary, one-hot encoding
  rnn.py             forward pass, BPTT, clipping, gradient checks
  losses.py          cross-entropy loss
  optimizers.py      SGD and Adam
  training.py        train/eval loop
  sampling.py        autoregressive generation
  utils.py           checkpoints, plotting, sequence windows
experiments/
  RESULTS.md         measured experiment log
  sequence_length.py
  optimizer_comparison.py
  gradient_analysis.py
tests/
  test_*.py
```

## Limitations

This is intentionally a vanilla RNN. It can learn local character patterns, but
it struggles with long-range coherence because gradients repeatedly pass through
`Waa` and `tanh`. Gradient clipping controls exploding gradients, but it does
not solve long-term memory. LSTMs, GRUs, and Transformers address these issues
with stronger mechanisms.

## LinkedIn Summary

I built a character-level Shakespeare language model from scratch using NumPy:
manual BPTT, gradient clipping, Adam, checkpoint metadata, validation tracking,
temperature sampling, and experiment logging. The model moved from random
character predictions at about `4.17` loss/char to about `2.13` validation
loss/char with a 128-unit vanilla RNN.
