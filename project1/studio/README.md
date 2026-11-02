# Studio code (excerpt)

The code all P1 runs were made with. It is an excerpt of my own experiment studio (a model designer, trainer and run browser
that I also use for other projects); I added ART as a text "source" and a supervised trainer there. The studio core (graph
compiler, storage, web server) is not included, so this folder does not run on its own: use `../standalone/` to retrain the final
models. Each run's `config.json` in `../experiments/` and `../final/` holds the complete design, preprocessing and training settings.

- `xcore/text.py`: preprocessing pipeline (switchable steps: normalize, lowercase, GloVe-style tokenizer, punctuation / stopword
  removal with a keep-list, Porter stemming, WordNet lemmatization), memory-mapped GloVe 6B, TF-IDF weights, vocabulary and encoding,
  own text features from formulas (e.g. `cos(h, o1) + cos(h, o2)`)
- `xcore/sources/art.py`: the ART source. Each hypothesis gets a score from (O1, O2, h) with the same weights for h1 and h2;
  templates for Arch 1 (mean / mean+max / pair features / all-in-one) and Arch 2 (GRU / LSTM, optional bidirectional)
- `xcore/sources/art_ops.py`: blocks: `rnn` (LSTM/GRU, packed sequences), `seq_join` (O1 → h → O2 with markers and part ids),
  `pair_features`, score / class outputs
- `xcore/train_supervised.py`: trainer: cross-entropy, Adam/AdamW/SGD, early stopping on validation loss, grad clipping, seeds,
  optional GPU (`ART_DEVICE=cuda`), W&B logging; `--test` evaluates the best epoch on the test split once (refuses a second time)
- `xcore/art_api.py`, `static/js/text*.js`: inspection UI (preprocessing preview, word neighbours, error browser, calibration)
- `tools/art_prepare.py`: downloads the course splits and GloVe, converts them to JSONL / numpy
- `tools/art_wandb_import.py`: mirrors runs trained without W&B (on the GPU machine) to W&B afterwards
- `tests/check_art.js`: headless UI smoke test
- `designs/`: the model designs with their version history
