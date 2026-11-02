# NLP HS26 – Abductive Commonsense Reasoning (ART)

HSLU NLP course projects (HS26, A. Marfurt).

Task: given two observations (O1, O2), pick the more plausible of two hypotheses (h1, h2).
Dataset: [amarfurt/art-nlp-hs26](https://huggingface.co/datasets/amarfurt/art-nlp-hs26) (train 16,801 / validation 1,000 / test 1,532).

- `project1/` – word embeddings + MLP, and frozen embeddings + 2-layer GRU (done, see below)
- `project2/` – Transformers (randomly initialized, pretrained, LLM)
- `EXPECTATIONS.md` – what I expected before the experiments, and (since 02.11.2026) what came out

Experiment tracking: Weights & Biases, project `sascha_ai/art-p1`, [saved P1 view](https://forge.coreweave.com/wandb/sascha_ai/art-p1?nw=5rz8ixg85cb).

## Project 1 at a glance

Final models, chosen on the validation split (lowest mean validation loss over 3 seeds), then tested **once** on the test split
(plan fixed before testing: [`project1/final/TEST_PLAN.md`](project1/final/TEST_PLAN.md)). Mean over 3 seeds, accuracy in %.

| | Model | Val acc | Val loss | **Test acc** | Test loss |
|---|---|---|---|---|---|
| Arch 1 | mean GloVe 300d (stopwords removed) of O1, O2, h (900 numbers) → MLP 512 ReLU, dropout 0.2; Adam lr 3e-4 | 71.5 | 0.555 | **53.1** | 0.98 |
| Arch 2 | O1 → h → O2 as one sequence (frozen GloVe 300d, part ids) → 2-layer GRU 128 → MLP 128 ReLU; Adam lr 3e-3, batch 64 | 70.1 | 0.561 | **53.5** | 0.87 |
| Chance | | 50 | 0.693 | 50 | 0.693 |

The test split is much harder than validation: validation is drawn like the training data, the test split was adversarially
filtered (the wrong hypothesis was picked to fool simple models). The models learned surface cues that work on validation and
not on test, e.g. "pick the shorter hypothesis" is right 57 % of the time on validation and 50 % on test.
Details, all ~190 runs and how to reproduce: [`project1/README.md`](project1/README.md).


## How the experiments were made: AI Experiment Studio

I did not write a training script per experiment. The models are built as graphs in my own experiment studio: pick the blocks,
switch preprocessing steps on/off, set the training options; the studio checks the shapes, generates the code from the graph,
trains, and logs every run (locally and to W&B). Each run's `config.json` is the full design.
The Studio itself was written by my coding agent env in OpenClaw, with mostly Claude. Instead of changing the code, i can change the settings and composition of the model within that studio and it automatically creates the code output and training runs for it. That also allows me to quickly make a few experiments.

Arch 1 final as a graph (O1, O2, h → mean over words → join, 900 numbers → MLP 512 → score):

![Arch 1 final in the studio](docs/studio-arch1-model.png)

The preprocessing, step by step on a real example (stopwords crossed out), and what the model receives:

![Inputs page](docs/studio-arch1-inputs.png)

The code generated from the graph, next to the raw design:

![Code page](docs/studio-arch1-code.png)

Arch 2 final (O1 → h → O2 joined into one sequence → 2-layer GRU → MLP → score): [screenshot](docs/studio-arch2-model.png)

## Folder structure

```
docs/           screenshots of the studio
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

