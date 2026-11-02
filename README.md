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
| Arch 1 | mean GloVe 300d (stopwords removed) of O1, O2, h + 1 similarity feature → MLP 512 ReLU, dropout 0.2; Adam lr 3e-4 | 71.5 | 0.555 | **53.1** | 0.98 |
| Arch 2 | O1 → h → O2 as one sequence (frozen GloVe 300d, part ids) → 2-layer GRU 128 → MLP 128 ReLU; Adam lr 3e-3, batch 64 | 70.1 | 0.561 | **53.5** | 0.87 |
| Chance | | 50 | 0.693 | 50 | 0.693 |

The test split is much harder than validation: validation is drawn like the training data, the test split was adversarially
filtered (the wrong hypothesis was picked to fool simple models). The models learned surface cues that work on validation and
not on test, e.g. "pick the shorter hypothesis" is right 57 % of the time on validation and 50 % on test.
Details, all ~190 runs and how to reproduce: [`project1/README.md`](project1/README.md).
