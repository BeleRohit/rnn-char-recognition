# Character-Level Language Model from Scratch

This project builds a character-level language model with a vanilla recurrent
neural network implemented in NumPy.

## What Is A Character-Level Language Model?

A character-level language model learns the probability of the next character
given the characters that came before it. If the text begins with `hell`, the
model tries to assign high probability to likely next characters such as `o`.

The model generates text one character at a time. Each newly generated character
is fed back into the model as the next input.

## How The Dataset Is Constructed

Raw text is treated as one long sequence of characters. For a sequence
`c0, c1, c2, ..., cT`, the inputs are:

```text
c0, c1, c2, ..., c(T-1)
```

The targets are shifted one step to the right:

```text
c1, c2, c3, ..., cT
```

So the learning problem at each time step is: predict the next character.

## Why Character Encoding Is Needed

Neural networks operate on numbers, not directly on text. Each unique character
is assigned an integer id, and each id can be represented as a one-hot vector.

For a vocabulary size `n_x = 4`, the character with id `2` is represented as:

```text
[0, 0, 1, 0]^T
```

This vector is `x<t>`, the input at time step `t`.

## Hidden State Intuition

The hidden state `a<t>` is the RNN's memory after reading characters up through
time step `t`. It is not a perfect memory of the full sequence. Instead, it is a
learned summary that helps predict the next character.

For example, after reading `qu`, the hidden state may store information that
makes `e`, `i`, or another vowel more likely than an unrelated character.

## RNN Architecture And Dimensions

The vanilla RNN equations are:

```text
a<t> = tanh(Wax x<t> + Waa a<t-1> + ba)
z<t> = Wya a<t> + by
y_hat<t> = softmax(z<t>)
```

Symbols:

- `n_x`: vocabulary size, also the input dimension.
- `n_a`: hidden state dimension.
- `n_y`: output vocabulary size. For character prediction, usually `n_y = n_x`.
- `T`: sequence length.
- `x<t>`: one-hot input vector at time `t`, shape `(n_x, 1)`.
- `a<t>`: hidden state at time `t`, shape `(n_a, 1)`.
- `y_hat<t>`: output probability vector at time `t`, shape `(n_y, 1)`.
- `Wax`: input-to-hidden weights, shape `(n_a, n_x)`.
- `Waa`: hidden-to-hidden weights, shape `(n_a, n_a)`.
- `Wya`: hidden-to-output weights, shape `(n_y, n_a)`.
- `ba`: hidden bias, shape `(n_a, 1)`.
- `by`: output bias, shape `(n_y, 1)`.

## Implemented Components

Core implementation:

- Text loading
- Vocabulary construction
- Character-to-index and index-to-character mappings
- One-hot encoding
- One-step-shifted input/target sequence creation
- Parameter initialization
- Numerically stable softmax
- Single RNN cell forward propagation
- Full sequence forward propagation
- Softmax cross-entropy loss
- Backpropagation Through Time
- Gradient accumulation for shared recurrent weights
- Elementwise gradient clipping
- Numerical gradient checking
- SGD updates
- Adam updates
- Training loop with loss and gradient norm tracking
- Model checkpoint save/load
- Temperature-controlled autoregressive sampling
- Experiment scripts for sequence length, optimizer comparison, and gradient behavior

## BPTT

The loss over a sequence is:

```text
L = -sum_t log(y_hat<t>[target<t>])
```

At each time step, the output gradient is:

```text
dz_y<t> = y_hat<t> - one_hot(target<t>)
```

The hidden state gradient has two sources:

```text
da<t> = Wya^T dz_y<t> + gradient from a<t+1>
```

The second term is what makes this Backpropagation Through Time. A hidden state
affects the current prediction and also affects future hidden states through
`Waa`.

Because the same matrices `Wax`, `Waa`, and `Wya` are reused at every time step,
their gradients are accumulated across the whole sequence.

## Training

Run a small training job:

```bash
python main.py --data data/shakespeare.txt --epochs 5 --hidden-size 64 --sequence-length 30
```

Generate text after training:

```bash
python main.py \
  --data data/shakespeare.txt \
  --epochs 5 \
  --sample-start T \
  --sample-length 300 \
  --temperature 0.8
```

Training prints the vocabulary size, number of update steps, initial loss, final
loss, and checkpoint path.

## Sampling

Generation is autoregressive:

1. Encode the current character as `x<t>`.
2. Compute `a<t>` and `y_hat<t>`.
3. Apply temperature scaling.
4. Sample the next character id.
5. Feed that sampled character into the next step.

Lower temperatures make samples more conservative. Higher temperatures make
samples more random.

## Experiments

The experiment scripts run measurements and print actual observed values. They
do not contain pre-filled results.

```bash
python experiments/sequence_length.py --data data/shakespeare.txt
python experiments/optimizer_comparison.py --data data/shakespeare.txt
python experiments/gradient_analysis.py --data data/shakespeare.txt
```

Record measured findings in `experiments/RESULTS.md`.

## Limitations Of Vanilla RNNs

Vanilla RNNs can learn short-range character patterns, but they struggle with
long-range dependencies because gradients repeatedly pass through recurrent
matrix multiplications and `tanh` derivatives. This can cause vanishing or
exploding gradients. Gradient clipping helps with exploding gradients, but it
does not fully solve long-term memory.

Modern sequence models often use LSTMs, GRUs, or Transformers to address these
limitations.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

Current verification:

```text
24 passed
```
