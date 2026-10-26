# Project 1: work-in-progress snapshot (19.10.2026)

These files are an excerpt of my own experiment studio (a model designer, trainer and run browser
that I also use for other projects). I added ART as a text dataset and a supervised trainer
there. This folder contains only the ART-specific parts, so it does not run on its own: the
studio core (graph compiler, storage, web server) is not included. A standalone version for the
submission will follow in `project1/`.

## What is here

- `xcore/text.py`: preprocessing pipeline (switchable steps, GloVe-style tokenizer), memory-mapped GloVe 6B, vocabulary and encoding, small text features (word overlap with O2, negations, cosine similarity)
- `xcore/sources/art.py`: the ART source. Each hypothesis gets a score from (O1, O2, h), with the same weights for h1 and h2; templates for Arch 1 (mean / mean+max pooling + 2-layer ReLU MLP) and Arch 2 (GRU / LSTM)
- `xcore/sources/art_ops.py`: blocks: `rnn` (LSTM/GRU, packed sequences), `seq_join` (O1 → h → O2 with markers), `pair_features`, score/class outputs
- `xcore/train_supervised.py`: trainer: cross-entropy, Adam/AdamW/SGD, early stopping on validation, grad clipping, seeds, W&B logging; `--test` evaluates the best epoch on the test split once
- `xcore/art_api.py`, `static/js/text*.js`: inspection UI (preprocessing preview, word neighbours, error browser, calibration)
- `tools/art_prepare.py`: downloads the course splits and GloVe, converts them to JSONL / numpy
- `tests/check_art.js`: headless UI smoke test
- `designs/`: model designs with their version history (Arch 1 v1–v5, Arch 2 GRU v1)
- `runs/`: config, per-step metrics, final status and validation predictions of the first runs (checkpoints are left out)

## First results (Arch 1, validation split, no test runs yet)

| Run | Pooling | Best val acc | W&B |
|---|---|---|---|
| `nlp-p1-arch-1-mean-glove-mlp-v3-s0-basel` | mean | 69.0 % (epoch 15) | [zpcf849u](https://forge.coreweave.com/wandb/sascha_ai/art-p1/runs/zpcf849u) |
| `arch-1-meanmax` | mean+max | 62.4 % (epoch 5) | [gucty9b9](https://forge.coreweave.com/wandb/sascha_ai/art-p1/runs/gucty9b9) |
| `nlp-meanmaxmoreepochs` | mean+max, more patience | 63.4 % (epoch 13) | [dlhwj09t](https://forge.coreweave.com/wandb/sascha_ai/art-p1/runs/dlhwj09t) |

Checks so far: with shuffled training labels the model gets 45 % on validation (chance level), so there is no label leak.
A first smoke run showed that the test split behaves differently from validation: surface cues
that gave about 60 % on validation gave about 50 % on test. Validation numbers probably overestimate.

Next: Arch 2 (frozen GloVe + 2-layer GRU/LSTM), one test run per final model, standalone code.
