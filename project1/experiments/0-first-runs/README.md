# First runs (19.10.2026, before the systematic experiments)

One folder per setting, one subfolder per seed with `config.json` (full design + preprocessing + training settings) and `result.json` (best epoch, validation, test if tested).
Sorted by mean validation loss (lower is better). Accuracy in %.

- `nlp-p1-arch-1-mean-glove-mlp-v3-s0-basel`: val loss 0.5803, val acc 69.0 (69.0, 1 seed). 
- `arch-2-gru-v1`: val loss 0.5946, val acc 68.3 (68.3, 1 seed). first Arch 2 run: 2-layer GRU 128, frozen GloVe, O1->h->O2
- `nlp-meanmaxmoreepochs`: val loss 0.6288, val acc 63.4 (63.4, 1 seed). 
- `arch-1-meanmax`: val loss 0.6411, val acc 62.4 (62.4, 1 seed). 
