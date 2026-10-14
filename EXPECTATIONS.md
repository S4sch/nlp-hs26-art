# Expectations

Written on 14.10.2026
Metric: test accuracy. Random baseline: 50%.

## Why this task is hard
- The dataset was filtered to keep only pairs that BERT could not solve, and even fine-tuned BERT did not do well on it.
- h- was made by editing up to 5 words of h+, so the two hypotheses are often very similar.
- It takes commonsense and cause/time reasoning, which static word embeddings don't capture well.

## Project 1

### Architecture 1: word embeddings + 2-layer ReLU classifier
**Expected test accuracy: 55%**

Static word vectors carry no context, so the classifier can mostly use surface cues
(like for example word overlap with O2, negations). If BERT struggled with this data, this model will likely land
only a little above chance.

### Architecture 2: frozen embeddings + 2-layer LSTM/GRU + 2-layer ReLU classifier
**Expected test accuracy: 59%**

The RNN reads the words in order, so it can use word order and context within the story
(and therefore kind of understand what "never" refers to, or what happens before/after). But the embeddings are still static and the model
has no real commonsense knowledge. On top of that, the training set is small (about 16.8k examples) for a
2-layer RNN with far more parameters than the classifier in architecture 1, so there is a fair chance it will overfit.
So i only expect a moderate gain over architecture 1.
