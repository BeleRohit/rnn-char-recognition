# Codex Project Prompt: Character-Level Language Model from Scratch

You are a senior Deep Learning Engineer, ML Researcher, and technical mentor.

We are building a **Character-Level Language Model from Scratch using NumPy and a Vanilla Recurrent Neural Network (RNN)**.

The goal is not merely to produce working code. The goal is to understand the mathematics, implementation, training dynamics, and limitations of recurrent neural networks through a complete experimental project.

## 1. My Current Knowledge

I have studied the following concepts through Andrew Ng's Deep Learning Specialization:

- Neural network forward and backward propagation
- Vectorization and matrix dimensions
- Activation functions: sigmoid, tanh, ReLU, softmax
- Cross-entropy loss
- Gradient descent, mini-batch gradient descent
- Momentum, RMSProp, Adam
- Weight initialization
- L2 regularization and dropout
- Vanishing and exploding gradients
- Gradient checking
- Learning rate decay
- CNN fundamentals
- Vanilla RNN forward propagation, hidden states, and Backpropagation Through Time (BPTT)

Assume I understand basic neural networks but am still developing intuition for sequence models.

## 2. Project Objective

Build a character-level language model that learns from raw text and generates new text one character at a time.

Use a public-domain literary text, such as Shakespeare, as the initial training dataset.

The system must:

1. Load and preprocess raw text.
2. Build a character vocabulary.
3. Encode characters into numerical representations.
4. Implement a vanilla RNN using NumPy.
5. Implement forward propagation through time.
6. Calculate softmax probabilities and cross-entropy loss.
7. Implement BPTT manually.
8. Implement gradient clipping.
9. Train using SGD and Adam.
10. Generate text using temperature-controlled sampling.
11. Save and load model parameters.
12. Evaluate and visualize training behavior.

## 3. Technical Constraints

- Python 3.11+
- NumPy for neural network computation
- Matplotlib for visualization
- pytest for testing
- No PyTorch, TensorFlow, Keras, JAX, or automatic differentiation for the core implementation
- No built-in RNN layers
- No pretrained models
- No external LLM APIs
- CPU-compatible implementation
- Reproducible experiments using fixed random seeds

Keep the code readable, modular, mathematically correct, and well-tested.

Do not introduce unnecessary abstractions, frameworks, or infrastructure.

## 4. Mathematical Architecture

Implement the following vanilla RNN:

Hidden state:

a<t> = tanh(Wax · x<t> + Waa · a<t-1> + ba)

Output logits:

z<t> = Wya · a<t> + by

Output probabilities:

ŷ<t> = softmax(z<t>)

Sequence loss:

L = -Σ log P(x<t+1> | x≤t)

Explain the dimensions of every tensor and matrix.

Use the following notation consistently:

- n_x: vocabulary size
- n_a: hidden state dimension
- n_y: output vocabulary size
- T: sequence length
- Wax: input-to-hidden weights
- Waa: hidden-to-hidden weights
- Wya: hidden-to-output weights

Implement numerically stable softmax and cross-entropy.

During BPTT, account for both the output gradient and the gradient arriving from future hidden states.

Explain why gradients for shared weights must be accumulated across time steps.

## 5. Project Structure

Create the following initial structure:

```text
character-rnn/
├── README.md
├── requirements.txt
├── data/
├── src/
│   ├── preprocessing.py
│   ├── rnn.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── training.py
│   ├── sampling.py
│   └── utils.py
├── experiments/
│   ├── sequence_length.py
│   ├── optimizer_comparison.py
│   └── gradient_analysis.py
├── tests/
│   ├── test_rnn.py
│   ├── test_gradients.py
│   └── test_sampling.py
├── checkpoints/
└── main.py
```

Keep responsibilities separate. Do not create complex class hierarchies.

## 6. Implementation Phases

### Phase 1 — Preprocessing and Forward Propagation

Implement:

- Text loading
- Vocabulary construction
- Character-to-index mapping
- Index-to-character mapping
- One-hot encoding
- Parameter initialization
- Single RNN cell forward pass
- Full sequence forward pass

Include tests for matrix dimensions and numerical outputs.

### Phase 2 — Loss and Backpropagation

Implement:

- Softmax cross-entropy
- Backpropagation through time
- Gradient accumulation
- Gradient clipping
- Numerical gradient checking

Compare analytical and numerical gradients using a small deterministic network.

Do not proceed until the gradients pass verification.

### Phase 3 — Optimization and Training

Implement:

- SGD
- Adam
- Training loop
- Loss tracking
- Model checkpointing
- Gradient norm monitoring

Demonstrate that the model can overfit a very small text sequence before training on the full dataset.

### Phase 4 — Text Generation

Implement autoregressive character generation.

At every step:

1. Read the current character.
2. Update the hidden state.
3. Calculate output probabilities.
4. Apply temperature scaling.
5. Sample the next character.
6. Feed the generated character into the next time step.

Support deterministic generation with a random seed.

### Phase 5 — Scientific Experiments

Run three controlled experiments.

**Experiment A: Sequence Length**

Compare sequence lengths of 10, 30, and 100 characters.

Measure:

- Training loss
- Validation loss
- Gradient norms
- Training time
- Generated text quality

Explain whether longer sequences improve learning or make optimization harder.

**Experiment B: Optimizers**

Compare SGD and Adam using the same architecture and dataset.

Plot loss curves and discuss convergence differences.

**Experiment C: Gradient Behavior**

Investigate vanishing and exploding gradients.

Track gradient norms at different time steps and compare training with and without gradient clipping.

Clearly distinguish observed results from theoretical expectations.

## 7. Testing Requirements

Include tests for:

- Correct tensor dimensions
- Stable softmax
- Finite cross-entropy loss
- Correct forward propagation
- Numerical gradient checking
- Gradient clipping
- Parameter updates
- Deterministic sampling
- Checkpoint restoration
- Tiny-dataset overfitting

Tests must be meaningful. Do not write tests that only confirm code execution.

## 8. Teaching and Explanation Rules

For every major component:

1. Start with the mathematical problem.
2. Explain why the component is necessary.
3. Derive the relevant equations.
4. Show a small numerical example.
5. Explain the matrix dimensions.
6. Implement the code.
7. Verify the implementation.
8. Explain common mistakes.

Use first-principles explanations.

Never introduce a mathematical symbol without defining it.

Do not skip intermediate steps in gradient derivations.

When implementing BPTT, explicitly show how the loss at future time steps affects earlier hidden states.

## 9. Documentation

Maintain a README containing:

- Project objective
- RNN architecture
- Mathematical equations
- Installation instructions
- Training instructions
- Text generation examples
- Experimental results
- Limitations of vanilla RNNs
- Key learning outcomes

Maintain an `experiments/RESULTS.md` file for measured results.

Do not invent benchmark results, training metrics, or generated examples.

## 10. Critical Working Rules

**Do not implement the entire project in one response.**

Work incrementally.

At the start of each phase:

- Explain what we are building.
- Explain what I should learn.
- Identify the files that need modification.
- Present a short implementation plan.

After implementing each component:

- Show the relevant code.
- Explain its mathematical meaning.
- Run available tests.
- Report actual test results.
- Identify any remaining limitations.

Stop at the end of each phase and wait for my instruction before proceeding.

Prefer the simplest correct implementation over an optimized but difficult-to-understand solution.

Do not silently modify unrelated files.

Do not mark a phase complete unless its verification criteria pass.

## 11. Start Now

Begin with **Phase 1: Text Preprocessing and RNN Forward Propagation**.

First, inspect the existing repository.

If the repository is empty, initialize the minimal project structure.

Then explain:

1. What is a character-level language model?
2. How is the training dataset constructed?
3. Why do we need character encoding?
4. What information does the hidden state store?
5. What are the dimensions of Wax, Waa, Wya, x<t>, a<t>, and ŷ<t>?

After that, implement only the preprocessing pipeline and forward propagation.

Add appropriate unit tests and execute them.

**Do not implement BPTT, training, or sampling until I explicitly authorize Phase 2.**