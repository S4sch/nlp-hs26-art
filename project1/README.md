# Project 1: word embeddings and RNNs on ART

Two architectures from the project description, all with frozen GloVe 6B 300d vectors:

1. **Arch 1:** each sentence becomes the mean of its word vectors; [O1, O2, h] (900 numbers) go through a 2-layer
   classifier (ReLU) that gives h a score. h1 and h2 are scored with the same weights; softmax over the two scores. Only the
   classifier is trained.
2. **Arch 2:** O1 → h → O2 are read as one sequence (an all-zero vector between the sentences, plus 3 "part id" numbers per word
   saying which sentence it comes from) by a 2-layer GRU; its last state goes through a 2-layer ReLU classifier → score for h.
   Embeddings frozen, GRU + classifier trained.

Training: cross-entropy over the two scores, Adam, gradient clipping 1, up to 30 epochs, early stopping when the validation loss
has not improved for 5 epochs; the weights of the best epoch (lowest validation loss) are kept. Every important setting was run
with 3 seeds (some with 5), and settings are compared by their **mean** over the seeds.

## Folder structure

```
final/          ⭐ the two final models: config, weights (best.pt), validation + test predictions, results; TEST_PLAN.md
experiments/    every run (config.json + result.json), sorted by question; a README per folder with all results
  0-first-runs/      first tries (19.10)
  1-preprocessing/   stopwords, punctuation, stemming, lemmatization, TF-IDF, GloVe size
  2-architecture/    pooling, pair features, all-in-one, GRU / BiGRU / BiLSTM, part ids
  3-tuning/          learning rate, hidden size, dropout, batch size (grid search)
results/        all_runs.csv (one row per run), settings.csv (one row per setting, mean over seeds)
standalone/     a single-file PyTorch version of the two final models (runs on its own)
studio/         the code the runs were made with (excerpt of my experiment studio, see below)
```

## Final models and test result

Selection rule (fixed before testing): lowest mean validation loss over 3 seeds; if two settings differ by less than the seed
spread, take the smaller model. Then each final model was evaluated **once** on the test split. See `final/TEST_PLAN.md`.

| | Setting | Val acc (3 seeds) | Val loss | **Test acc** | Test loss |
|---|---|---|---|---|---|
| Arch 1 | `n4-a1-lr0.0003-h512-d0.2` (461,825 weights) | 71.5 (71.3–71.6) | 0.555 | **53.1** (52.9–53.3) | 0.98 |
| Arch 2 | `n4-a2-lr0.003-h128-b64` (281,985 weights) | 70.1 (68.8–71.6) | 0.561 | **53.5** (52.1–54.8) | 0.87 |
| Ensembles | average of the 3 seeds / Arch 1 + Arch 2 | 71.6–73.5 | 0.528–0.552 | 52.5–53.9 | |

**Validation ~71 %, test ~53 %.** I checked that this is not a bug (same evaluation code gives 71.8 % on validation; labels
balanced; no story shared between splits). The test split (1,532 examples) is adversarially filtered: the wrong hypothesis was
chosen so that simple models fail. Validation is not. Example: "always pick the shorter hypothesis" is right 57.1 % of the time
on validation but 49.9 % on test. The models learned such cues; on the test split they are confidently wrong (test loss 0.87–0.98,
worse than guessing, 0.693). Ensembles helped on validation (+2 points for the GRU) but not on test: all models share the same
blind spot.

### Analysis after the final test: did the validation gains carry over?

After the final test I also tested the other key settings once (list fixed before running it, see `final/TEST_PLAN.md`; the
final models were **not** re-selected). Mean over seeds, accuracy in %:

| Setting | Val acc | Test acc | Test loss |
|---|---|---|---|
| Arch 1 baseline (`n1-a1-mean`) | 68.7 | 52.4 (51.6–53.1) | 1.00 |
| Arch 1 no stopwords (`n1-a1-nostop`) | 70.8 | 52.5 (52.0–53.0) | 1.00 |
| Arch 1 stemming (`n2-a1-nostop-stem`) | 65.8 | 53.2 (53.1–53.3) | 0.83 |
| **Arch 1 final** | 71.5 | **53.1** | 0.98 |
| Arch 2 GRU start (`n1-a2-gru`, 2 seeds) | 69.0 | 52.9 | 0.90 |
| Arch 2 GRU + part ids (`n3t-a2-gru-nostop-partids`, 5 seeds) | 69.8 | 52.8 (52.4–53.1) | 0.93 |
| **Arch 2 final** | 70.1 | **53.5** | 0.87 |
| Arch 2 BiLSTM (`n1-a2-bilstm`, 1 seed) | 52.6 | 49.9 | 0.693 |

The +2.8 points on validation from baseline to final became +0.7 on test; all trained models land between 52.4 and 53.5
(one standard deviation of pure guessing on 1,532 examples is 1.3 points). So most of what improved validation was fitting cues
that the filtered test split removes. Stemming, the worst setting on validation, is as good as the final on test and the least
over-confident (lowest test loss): with less detail in the input it learned fewer of these cues.

## What was tried (mean validation results; full list in `experiments/*/README.md` and `results/settings.csv`)

Preprocessing (Arch 1, 3 seeds each):
- Baseline (lowercase, GloVe tokenizer): 68.7 %, loss 0.573
- **Remove stopwords** (keeping negations and time / cause words: not, never, before, after, because, but, …): **70.8 %, loss 0.558** (+2.1, best single change)
- + remove punctuation: 70.9 % (no effect) · TF-IDF weighted mean: 70.0 % alone, 70.7 % on top of stopword removal (no gain)
- Lemmatization: 69.6 % (−1.2) · **Stemming: 65.8 % (−5.0)**: 10.6 % of all tokens became stems without a GloVe vector
- GloVe 50d / 100d / 200d / 300d: 66.6 / 68.3 / 70.7 / 70.8 %

Architecture:
- Arch 1 mean+max pooling: 62–63 % (the max values are not normalised, unstable training)
- Arch 1 pair features (InferSent-style h·O, |h−O|, cos): 69.1 %; on means only 71.0 % (on par with the final, higher loss)
- Arch 1 all-in-one (both hypotheses in one input → 2 classes): 63.5 %; with swap augmentation 68.2 %
- Arch 2 GRU 69.0 % · BiGRU 69.7 % · **BiLSTM stuck at chance** (52.6 %, loss = ln 2 for all epochs)
- GRU part ids: 69.8 % vs 69.3 % without (5 seeds each), lower validation loss on all 5 seeds

Tuning (grid search, 3 seeds per point, on a GPU):
- Arch 1: lr {3e-4, 1e-3, 3e-3} × hidden {128, 256, 512} × dropout {0.2, 0.5}; the best point was at the edge (lr 3e-4, hidden
  512), so the range was extended (lr 1e-4, hidden 1024): flat, no further gain
- Arch 2: lr {3e-4, 1e-3, 3e-3} × GRU size {64, 128, 256} × batch {64, 256}, extended to lr 1e-2: lr 3e-3, batch 64 best; batch 256 worse everywhere
- The GRU overfits fast: best epoch 3–4, then train accuracy goes to 90 %+ while the validation loss rises

## How to reproduce

- `standalone/art_p1.py` trains and tests the two final models on its own (PyTorch + numpy); see `standalone/README.md`.
- The runs themselves were made with the code in `studio/` (my experiment studio, where ART is one "source"). That folder is an
  excerpt: the studio core (graph compiler, storage, web UI) is not included, so it does not run on its own. Each run's
  `config.json` contains the complete design (graph), preprocessing pipeline and training settings.
- Data: the course splits from Hugging Face and GloVe 6B 300d (`studio/tools/art_prepare.py` or `standalone/art_p1.py prepare`).

## Use of a coding agent

I used a coding agent (Claude, via OpenClaw) for this project. It wrote most of the code in `studio/` (ART source, preprocessing
pipeline, trainer, W&B logging) and `standalone/`, ran most of the experiment batches and helped analyse the results. The research
questions, the expectations (`../EXPECTATIONS.md`), the choice of experiments and the conclusions were discussed and decided by me;
some experiments I set up and ran myself in the studio (e.g. GRU without stopwords, GRU dropout 0.4).
