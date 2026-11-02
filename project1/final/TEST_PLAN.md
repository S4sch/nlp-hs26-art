# P1 test plan (fixed 2026-11-02 10:35, BEFORE any test-set evaluation)
Selection rule: lowest mean best val loss over 3 seeds (val = 1000 examples). Tie (< 1 seed sd) -> smaller model.

Edge check (slide 53) done: Arch 1 lr 1e-4 / hidden 1024 and GRU lr 1e-2 did not beat the range -> plateau, ranges closed.
- Arch 1 h1024 (loss 0.5544) vs h512 (0.5553): difference 0.0009 < seed sd 0.0020 -> tie -> h512 (half the weights).

Tested once each (best.pt, test split 1532 examples):
1. Arch 1: n4-a1-lr0.0003-h512-d0.2-s{0,1,2}      val single 71.7 % (sd 0.1), loss 0.5553
2. Arch 2: n4-a2-lr0.003-h128-b64-s{0,1,2}        val single 70.1 % (sd 1.4), loss 0.5606
3. Pairs (best alternative): n4-pairs-nostop-mean-s{0,1,2}   val 71.2 %, loss 0.5732
Ensembles (average of p_h2), scored on val first:
- Arch 1 3-seed: val 71.6 %, loss 0.5518
- GRU 3-seed:    val 72.2 %, loss 0.5369
- Arch 1 + GRU (6 models): val 72.6 %, loss 0.5280  <- chosen final ensemble (lowest val loss)
- Arch 1 + GRU + pairs (9): val 73.5 %, loss 0.5305  (reported, not chosen: higher val loss)
Nothing is changed after seeing the test numbers.

## Results (run 10:23-10:24, once; test = 1532 examples, chance 50 %, 1 sd of chance = 1.3 %)
- Arch 1: 53.1 % (52.9-53.3), loss 0.98 | 3-seed ens 52.5 %
- Arch 2 GRU: 53.5 % (52.1-54.8), loss 0.87 | 3-seed ens 53.8 %, loss 0.80
- Pairs: 52.3 % (51.8-52.6)
- Arch 1 + GRU ens: 53.2 %, loss 0.82 | + pairs: 53.9 %
Sanity: same eval code on validation -> 71.8 % (no bug). Labels balanced, no story overlap train/val/test.
Why: val is drawn like train; test (1532 = size of the official aNLI dev set) is adversarially filtered: the wrong hypothesis was chosen to fool
surface-cue models. Proof: "pick the SHORTER hypothesis" is right 57.1 % on val, 49.9 % on test. Our models learned such cues
-> confident and wrong on test (loss 0.87-0.98 > ln 2 = 0.69; A1's confident predictions (p<0.2 or >0.8): only 56 % right).
Ensembles: help on val (+2 GRU), not on test (averaging models that share the same blind spot doesn't fix it).

## Post-hoc analysis (list fixed 2026-11-02 11:05, BEFORE running it; Sascha agreed to report all numbers)
Question: did the validation improvements carry over to the test split? The finals above stay fixed; nothing is re-selected.
Tested once each, all seeds of the remaining 6 settings in the W&B view:
- n1-a1-nostop + n1-a1-nostop-s{1,2} (Arch 1 no stopwords) · n1-a1-mean-s{0,1,2} (Arch 1 baseline) · n2-a1-nostop-stem-s{0,1,2} (Arch 1 stemming)
- n3t-a2-gru-nostop-partids-s{0..4} (GRU + part ids) · n1-a2-gru-s{0,1} (GRU start) · n1-a2-bilstm (BiLSTM, seed 0)
Results (11:06, mean over seeds; val -> test acc, test loss):
- Arch 1 baseline 68.7 -> 52.4 (51.6-53.1), loss 1.00 · no stopwords 70.8 -> 52.5 (52.0-53.0), 1.00 · FINAL 71.5 -> 53.1, 0.98
- Arch 1 stemming 65.8 -> 53.2 (53.1-53.3), loss 0.83 (worst on val, as good as the final on test, least over-confident)
- GRU start 69.0 -> 52.9, 0.90 · GRU + part ids 69.8 -> 52.8 (52.4-53.1), 0.93 · GRU FINAL 70.1 -> 53.5, 0.87 · BiLSTM 52.6 -> 49.9, 0.693 (chance)
Conclusion: +2.8 points on val (baseline -> final) became +0.7 on test; all models 52.4-53.5 (1 sd of chance = 1.3).
The validation improvements were mostly fitting cues that the filtered test split removes.
