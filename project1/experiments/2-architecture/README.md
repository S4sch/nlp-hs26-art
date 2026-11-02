# Architecture variants: pooling, pair features, all-in-one, GRU / BiGRU / BiLSTM, part ids

One folder per setting, one subfolder per seed with `config.json` (full design + preprocessing + training settings) and `result.json` (best epoch, validation, test if tested).
Sorted by mean validation loss (lower is better). Accuracy in %.

- `n3-a2-gru-nostop-pool`: val loss 0.5685, val acc 68.9 (68.0-69.8, 2 seeds). Arch 2 GRU without stopwords, only change: mean+max over ALL GRU states instead of the last state
- `n3t-a2-gru-nostop-partids`: val loss 0.5711, val acc 69.8 (69.1-70.4, 5 seeds), **test 52.8**. Arch 2 GRU without stopwords, part ids ON - tower GPU (GTX 1060), 5-seed comparison
- `n4-pairs-nostop-mean`: val loss 0.5732, val acc 71.0 (70.6-71.3, 3 seeds), **test 52.2**. Arch 1+ without stopwords, comparisons on MEAN pooling only (no unnormalised max)
- `n1-a2-bigru`: val loss 0.5751, val acc 69.7 (69.7, 1 seed). Arch 2: bidirectional GRU (is the BiLSTM plateau about LSTM or about reading backwards?)
- `n4-pairs-nostop-mean-lr3e-4`: val loss 0.5786, val acc 71.1 (70.8-71.2, 3 seeds). Arch 1+ without stopwords, mean pooling, lr 3e-4 (4x more weights -> smaller steps)
- `n1-a2-gru`: val loss 0.5872, val acc 69.0 (68.3-69.7, 2 seeds), **test 52.9**. Arch 2 GRU again, patience 5 (reference)
- `n4-all-swap-nostop`: val loss 0.5913, val acc 68.2 (67.1-69.9, 3 seeds). Arch 1 all in one + swap augmentation, without stopwords
- `n4-pairs-nostop`: val loss 0.5954, val acc 69.1 (68.4-70.1, 3 seeds). Arch 1+ (mean+max + InferSent comparisons) without stopwords - fair second try
- `n1-a1-all-swap`: val loss 0.6016, val acc 68.2 (68.2, 1 seed). Arch 1 all in one + swap augmentation (h1<->h2 randomly each epoch)
- `n1-a1-pairs`: val loss 0.6093, val acc 68.4 (68.4, 1 seed). Arch 1+: mean+max + InferSent comparisons h.O, |h-O|, cos (Conneau et al. 2017)
- `n1-a1-all`: val loss 0.6459, val acc 63.5 (63.5, 1 seed). Arch 1, other I/O format: all in one [O1,O2,h1,h2] -> 2 classes
- `n1-a2-bilstm`: val loss 0.6931, val acc 52.6 (52.6, 1 seed), **test 49.9**. Arch 2 variant: 2-layer bidirectional LSTM reads O1 -> h -> O2
