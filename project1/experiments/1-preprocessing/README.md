# Preprocessing: stopwords, punctuation, stemming / lemmatization, TF-IDF weighting, GloVe size

One folder per setting, one subfolder per seed with `config.json` (full design + preprocessing + training settings) and `result.json` (best epoch, validation, test if tested).
Sorted by mean validation loss (lower is better). Accuracy in %.

- `n3-a2-gru-nostop`: val loss 0.5568, val acc 71.2 (71.2, 1 seed). Arch 2 GRU without stopwords (reference for the pooling runs)
- `n1-a1-nostop`: val loss 0.5580, val acc 70.8 (70.3-71.4, 3 seeds). Arch 1, only change: remove stopwords (keep negations / time words) -> do h1/h2 means separate more?
- `n3-a1-nostop-punct`: val loss 0.5606, val acc 70.9 (70.4-71.2, 3 seeds). Arch 1 without stopwords, only change: also remove punctuation
- `n3-a1-nostop-idf`: val loss 0.5664, val acc 70.7 (70.2-71.6, 3 seeds). Arch 1 without stopwords, only change: TF-IDF weighted mean of the word vectors (IDF from train sentences)
- `n3-a1-idf`: val loss 0.5690, val acc 70.0 (69.5-70.5, 3 seeds). Arch 1, only change: TF-IDF weighted mean of the word vectors (IDF from train sentences)
- `n1-a1-mean`: val loss 0.5728, val acc 68.7 (67.4-69.5, 3 seeds). Arch 1 baseline again, patience 5 (fair reference for tonight's runs)
- `n3-a1-nostop-d100`: val loss 0.5730, val acc 68.3 (68.3, 1 seed). Arch 1 without stopwords, only change: GloVe 100d instead of 300d
- `n3-a1-nostop-d200`: val loss 0.5730, val acc 70.7 (70.7, 1 seed). Arch 1 without stopwords, only change: GloVe 200d instead of 300d
- `n2-a1-nostop-lemm`: val loss 0.5742, val acc 69.6 (69.1-70.2, 3 seeds). Arch 1 without stopwords (best setup) + lemmatize (only change) - Sascha's experiment 02.11
- `n2-a2-gru-nostop`: val loss 0.5762, val acc 69.6 (69.6, 1 seed). Arch 2 GRU, only change: remove stopwords (keep negations / time words) - Sascha's experiment 02.11
- `n3t-a2-gru-nostop`: val loss 0.5785, val acc 69.3 (67.3-70.7, 5 seeds). Arch 2 GRU without stopwords, part ids off - tower GPU (GTX 1060), 5-seed comparison
- `n3-a1-nostop-d50`: val loss 0.6100, val acc 66.6 (66.6, 1 seed). Arch 1 without stopwords, only change: GloVe 50d instead of 300d
- `n2-a1-nostop-stem`: val loss 0.6150, val acc 65.8 (64.9-66.8, 3 seeds). Arch 1 without stopwords (best setup) + stem (only change) - Sascha's experiment 02.11
