# Experiment Results

Experiment scripts append measured results to this file automatically.

Run examples:

```bash
.venv/bin/python experiments/sequence_length.py --data data/shakespeare.txt
.venv/bin/python experiments/optimizer_comparison.py --data data/shakespeare.txt
.venv/bin/python experiments/gradient_analysis.py --data data/shakespeare.txt
```

## Experiment B: Optimizers

Run timestamp: 2026-10-13 16:46:50 IST

Metadata:
- `data`: `data/shakespeare.txt`
- `characters`: `1115394`
- `vocab_size`: `65`
- `epochs`: `3`
- `hidden_size`: `64`
- `sequence_length`: `30`

| optimizer | learning_rate | initial_loss | initial_loss_per_char | final_loss | final_loss_per_char | last_grad_norm | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sgd | 0.1 | 125.2280 | 4.1743 | 748.0235 | 24.9341 | 232.0557 | 152.48 |
| adam | 0.01 | 125.2280 | 4.1743 | 83.9587 | 2.7986 | 39.7736 | 154.19 |



## Manual Training Run: h128 seq50 continuation

- checkpoint: checkpoints/shakespeare-rnn-h128-test-seq50.npz
- final_train_loss_per_char: 1.7158
- final_val_loss_per_char: 2.1348
- total_sec: 88.3
- note: Improved validation loss from 2.1922 to 2.1348.