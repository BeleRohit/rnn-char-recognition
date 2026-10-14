PYTHON ?= .venv/bin/python
DATA ?= data/shakespeare.txt
CHECKPOINT ?= checkpoints/shakespeare-rnn-h128.npz

.PHONY: install test train-small sample sample-metadata experiments clean

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

train-small:
	$(PYTHON) main.py \
		--data $(DATA) \
		--epochs 1 \
		--hidden-size 32 \
		--sequence-length 30 \
		--learning-rate 0.01 \
		--optimizer adam \
		--checkpoint checkpoints/smoke.npz \
		--plot-path plots/smoke-loss.png \
		--progress-every 1000

sample:
	$(PYTHON) main.py \
		--data $(DATA) \
		--load-checkpoint $(CHECKPOINT) \
		--sample-only \
		--sample-start T \
		--sample-length 1000 \
		--temperature 0.7 \
		--seed 42

sample-metadata:
	$(PYTHON) main.py \
		--load-checkpoint $(CHECKPOINT) \
		--sample-only \
		--sample-start T \
		--sample-length 1000 \
		--temperature 0.7 \
		--seed 42

experiments:
	$(PYTHON) experiments/optimizer_comparison.py --data $(DATA)
	$(PYTHON) experiments/sequence_length.py --data $(DATA)
	$(PYTHON) experiments/gradient_analysis.py --data $(DATA)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
