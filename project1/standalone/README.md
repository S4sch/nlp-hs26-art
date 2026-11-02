# Standalone re-implementation of the two final models

`art_p1.py` is one plain PyTorch file that trains and evaluates the two final models of Project 1 **without** the experiment
studio. It was written with a coding agent (Claude) from the studio's code (preprocessing, word vectors, the graph blocks
that were used, the supervised trainer), and then checked against the studio (see [Verification](#verification)).

| | Arch 1 | Arch 2 |
|---|---|---|
| input | mean GloVe 300d vector of O1, O2, h (900 numbers) | words of O1, h, O2 as one sequence, all-zero marker word between the sentences, 3 part-id numbers per word (303 per word) |
| model | MLP 900 → 512 → 1, ReLU, dropout 0.2 | 2-layer GRU 128 (dropout 0.2 between the layers), last state → MLP 128 → 128 → 1, ReLU, dropout 0.2 |
| training | Adam lr 3e-4, batch 64 | Adam lr 3e-3, batch 64 |
| weights | 461,825 | 281,985 |

Both: frozen GloVe 6B 300d (unknown words = zero vector), preprocessing normalize → lowercase → GloVe-style tokenizer →
stopwords removed except a keep list (negations, *before / after / then / because / but / so / again / too / only / until / while*),
punctuation kept. h1 and h2 are scored with the same weights, cross-entropy over the two scores, gradient clipping 1, at most 30
epochs, early stopping when the validation loss has not improved for 5 epochs, the weights of the best epoch are kept.

## How to run

Needs Python 3 with `torch` and `numpy` (`pyarrow` only for `prepare`).

```bash
python art_p1.py prepare --data data/                        # dataset (amarfurt/art-nlp-hs26) + GloVe 6B (822 MB zip), converted
python art_p1.py train --arch 1 --seed 0 --data data/ --out runs/a1-s0
python art_p1.py train --arch 2 --seed 0 --data data/ --out runs/a2-s0
python art_p1.py evaluate --run runs/a1-s0 --data data/      # best checkpoint on the validation split
python art_p1.py test --run runs/a1-s0 --data data/          # best checkpoint on the test split: once
```

- `train` uses the final configuration of the chosen architecture; change it with `--lr --hidden --dropout --batch --epochs`
  (Arch 2: `--hidden` sets both the GRU and the MLP size, as in the grid search). Also `--patience`, `--device cuda`,
  `--threads` (default 2), `--minutes` (time limit, default 60).
- `test` writes the result into `result.json` and refuses to run a second time (`--force` overrides it): choosing between
  several test results would be tuning on the test set.
- `--wandb` (off by default) logs to Weights & Biases (`--wandb-project`, `--wandb-entity`); log in with `wandb login` first.
- `--similarity-feature` (Arch 1 only, **not** part of the final model, see below) also feeds the standardised
  `cos(h, o1) + cos(h, o2)` to the MLP.

A run directory holds `config.json`, `metrics.jsonl` (one line per epoch), `best.pt`, `val_predictions.json` and `result.json`.

## Verification

All checks used the training and validation splits only; the test split was not touched (the final models were already tested
once in the studio).

**Same computation as the studio** (studio code imported read-only, CPU):

- tokens of every train and validation sentence: identical; vocabulary (15,674 words, 752 without a GloVe vector) and vector
  table: identical;
- weight counts: 461,825 and 281,985, the same as the studio; the initial weights for a seed are identical, and the state-dict
  names are the same (studio checkpoints load after removing the `net.` prefix);
- the studio's best checkpoints give exactly the same scores in both implementations (largest difference 0.0);
- training: the studio's own trainer, run on the CPU, and `art_p1.py` give **bit-identical** train / validation loss and accuracy
  for the first 2 epochs of both architectures (seed 0);
- `prepare`: the jsonl files converted from the parquet files and the GloVe `.npy` / vocabulary converted from the zip are
  byte-identical to the ones the studio used.

**Full runs, compared with the studio** (best epoch by validation loss; validation accuracy / loss):

| | seed | studio (tower GPU) | `art_p1.py` (CPU, 2 threads) |
|---|---|---|---|
| Arch 1 | 0 | 71.6 % / 0.553, epoch 20 | 71.2 % / 0.556, epoch 21 |
| | 1 | 71.6 % / 0.557, epoch 18 | 71.8 % / 0.560, epoch 17 |
| | 2 | 71.3 % / 0.555, epoch 18 | 70.7 % / 0.556, epoch 18 |
| | mean | **71.5 %** / 0.555 | **71.2 %** / 0.557 |
| Arch 2 | 0 | 69.8 % / 0.560, epoch 3 | 70.1 % / 0.555, epoch 2 |
| | 1 | 68.8 % / 0.573, epoch 4 | 69.6 % / 0.572, epoch 2 |
| | 2 | 71.6 % / 0.549, epoch 3 | 68.2 % / 0.567, epoch 3 |
| | mean | **70.1 %** / 0.561 | **69.3 %** / 0.565 |

Run time on the Steam Deck CPU (2 threads): Arch 1 about 2 minutes per run (22-26 epochs), Arch 2 5-9 minutes (7-8 epochs).

## Differences (honest list)

- **GPU vs CPU.** The studio's final runs were trained on a GPU. The GPU draws different random numbers for dropout and adds
  numbers in a different order, so the same seed gives a different run on the CPU. That is why the full runs above differ by
  a few tenths of a percent per seed (Arch 2 seed 2: 3.4 points; with only 1,000 validation examples, one seed's accuracy
  varies by ±1-2 points anyway). On the CPU the two implementations are bit-identical, so these differences come from the
  hardware, not from the code. The means over 3 seeds are within 0.3 (Arch 1) and 0.8 (Arch 2) points.
- **Ties in Arch 1.** In 17 validation examples h1 and h2 contain the same words after stopword removal (e.g. "Kelly went to
  the hospital." / "The hospital went to Kelly."), so a mean of word vectors gives both the same score, up to rounding. Which
  one wins is decided by rounding: the studio's best Arch 1 checkpoint (seed 0) gives 71.6 % on the GPU and 71.8 % on the CPU
  (4 examples at p = 0.5 fall the other way; the loss is the same, 0.55323).
- **The `Higher_Similarity` feature was not used.** The Arch 1 design defines the text feature `cos(h, o1) + cos(h, o2)`, and
  the studio computed it, but the design's graph had no input block for features, so it never reached the model: the MLP
  has 900 inputs (461,825 weights). The re-implementation therefore leaves it out by default. `--similarity-feature` adds it
  (standardised with the train split's mean 1.384 and spread 0.186, the same numbers the studio computed; 462,337 weights);
  this variant was not part of the experiments and is not tuned.
- Left out (not used by the final models): the studio's other preprocessing steps, other blocks, the "all in one" format,
  the time-limited partial-data option, the live dashboard.
