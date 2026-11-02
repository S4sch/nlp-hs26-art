#!/usr/bin/env python3
"""NLP HS26 Project 1 (ART, abductive commonsense reasoning): the two final models, standalone.

A re-implementation of what the experiment studio did for the two final designs, in one plain PyTorch file:

    Arch 1  mean of frozen GloVe vectors of O1, O2 and h  ->  MLP (900 -> 512 -> 1) = a score for h
    Arch 2  the words of O1, h, O2 as one sequence (zero "marker" word between the sentences, 3 part-id numbers per word)
            -> 2-layer GRU (128) -> last state -> MLP (128 -> 128 -> 1) = a score for h

Both models score h1 and h2 with the same weights; a softmax over the two scores gives the answer (cross-entropy loss).

Usage
    python art_p1.py prepare --data data/                       # download the dataset + GloVe 6B, convert (once)
    python art_p1.py train --arch 1 --seed 0 --data data/ --out runs/a1-s0
    python art_p1.py train --arch 2 --seed 0 --data data/ --out runs/a2-s0
    python art_p1.py evaluate --run runs/a1-s0 --data data/     # best checkpoint on VALIDATION (as often as you like)
    python art_p1.py test --run runs/a1-s0 --data data/         # best checkpoint on TEST, once (refused a second time)

Data directory (made by `prepare`):
    train.jsonl / validation.jsonl / test.jsonl   rows of amarfurt/art-nlp-hs26 (observation_1/2, hypothesis_1/2, label 1|2)
    glove/6B.300.npy                              float32 [400000, 300], rows in the order of the original GloVe file
    glove/6B.vocab.txt                            one word per line (row i of the .npy)

Run directory (made by `train`): config.json, metrics.jsonl (one line per epoch), best.pt (weights of the epoch with the
lowest validation loss), val_predictions.json, result.json (best epoch, timing; the test result once it was run).
"""
import argparse
import json
import pathlib
import random
import re
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

SPLITS = ("train", "validation", "test")
FIELDS = ("o1", "o2", "h1", "h2")
DIM = 300  # GloVe 6B, 300 dimensions

# The final configurations (chosen on the validation split by a grid search over lr / hidden size / dropout / batch size).
DEFAULTS = {
    1: {"lr": 3e-4, "hidden": 512, "dropout": 0.2, "batch": 64, "epochs": 30},
    2: {"lr": 3e-3, "hidden": 128, "dropout": 0.2, "batch": 64, "epochs": 30},
}
PATIENCE, GRAD_CLIP, MINUTES = 5, 1.0, 60  # early stopping on the validation loss, gradient clipping, time limit (safety net)

HF_URL = "https://huggingface.co/datasets/amarfurt/art-nlp-hs26/resolve/main/anli/{split}-00000-of-00001.parquet"
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"


# ============================================================================================ preprocessing
# Pipeline of the final designs: normalize -> lowercase -> tokenize (GloVe style) -> remove stopwords (keep a list).
# Punctuation is NOT removed (the "remove punctuation" step was switched off).

STOPWORDS = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself
they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
shouldn wasn weren won wouldn n't 's 're 've 'll 'd 'm""".split())
# stopwords that carry meaning for this task (negation, time order, cause) and are kept
KEEP = set("not n't no never nothing nobody none nowhere neither nor without cannot before after then because but so again too only until while".split())


def normalize(s):
    """Curly quotes -> straight quotes, long dashes -> '-', runs of whitespace -> one space."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s):
    """Split like the Stanford tokenizer GloVe was trained with: punctuation separate, "didn't" -> "did" "n't", "she's" -> "she" "'s"."""
    s = re.sub(r"(?i)([a-z])n't\b", r"\1 n't", s)
    s = re.sub(r"(?i)(?<=[a-z])'(s|m|d|ll|re|ve)\b", r" '\1", s)
    return re.findall(r"n't|'(?:s|m|d|ll|re|ve)\b|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\sA-Za-z0-9]", s, flags=re.I)


def preprocess(sentence):
    toks = tokenize(normalize(str(sentence or "")).lower())
    return [t for t in toks if t not in STOPWORDS or t in KEEP]


def load_split(data, split):
    """Rows of a split as token lists; label 0 = h1 is the plausible hypothesis, 1 = h2."""
    rows = []
    with open(pathlib.Path(data) / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"o1": preprocess(r["observation_1"]), "o2": preprocess(r["observation_2"]),
                         "h1": preprocess(r["hypothesis_1"]), "h2": preprocess(r["hypothesis_2"]), "label": int(r["label"]) - 1})
    return rows


# ============================================================================================ word vectors
class Glove:
    """GloVe 6B 300d, memory-mapped (only the rows of words that are looked up are read from disk)."""

    def __init__(self, data):
        g = pathlib.Path(data) / "glove"
        self.M = np.load(g / f"6B.{DIM}.npy", mmap_mode="r")
        self.index = {w: i for i, w in enumerate((g / "6B.vocab.txt").read_text().split("\n")[:-1])}

    def row(self, w):
        r = self.index.get(w)
        return self.index.get(w.lower()) if r is None else r  # try the lowercase form too

    def vec(self, w):
        r = self.row(w)
        return None if r is None else np.asarray(self.M[r], dtype=np.float32)


class Vocab:
    """Word ids of the loaded splits. Row 0 = padding, row 1 = every unknown word (a vector of zeros). The vectors are frozen."""

    def __init__(self, rows_per_split, glove):
        words, self.unknown = set(), set()
        for rows in rows_per_split:
            for r in rows:
                for f in FIELDS:
                    for t in r[f]:
                        (words if glove.row(t) is not None else self.unknown).add(t)
        self.ids = {w: i + 2 for i, w in enumerate(sorted(words))}
        self.matrix = np.zeros((len(self.ids) + 2, DIM), dtype=np.float32)  # rows 0 and 1 stay zero
        for w, i in self.ids.items():
            self.matrix[i] = glove.vec(w)

    def encode(self, toks):
        return [self.ids.get(t, 1) for t in toks]  # unknown words keep their place (id 1, zero vector)


class Split:
    """One split as word ids (+ the optional similarity feature), and its batches."""

    def __init__(self, rows, vocab, feature=None):
        self.ids = {f: [vocab.encode(r[f]) for r in rows] for f in FIELDS}
        self.labels = np.array([r["label"] for r in rows], dtype=np.int64)
        self.feature = feature  # [n, 2] standardised feature of (h1, h2), or None
        self.n = len(rows)

    def batch(self, idx):
        """2 * len(idx) rows: first every example with h = h1, then every example with h = h2 (same O1 / O2)."""
        I = self.ids
        b = {"o1": pad([I["o1"][i] for i in idx] * 2), "o2": pad([I["o2"][i] for i in idx] * 2),
             "h": pad([I["h1"][i] for i in idx] + [I["h2"][i] for i in idx]), "y": torch.from_numpy(self.labels[idx].copy())}
        if self.feature is not None:
            b["feature"] = torch.from_numpy(np.concatenate([self.feature[idx, 0], self.feature[idx, 1]], 0)).unsqueeze(-1)
        return b


def pad(lists):
    """Id lists -> [B, longest] tensor, 0 = padding (at least one column, so empty sentences still work)."""
    out = np.zeros((len(lists), max([len(x) for x in lists] + [1])), dtype=np.int64)
    for i, x in enumerate(lists):
        out[i, :len(x)] = x
    return torch.from_numpy(out)


# ------------------------------------------------------------------------------ optional own feature (not in the final models)
def similarity_feature(rows, glove):
    """Higher_Similarity = cos(h, o1) + cos(h, o2) for h = h1 and h = h2 -> [n, 2]. Cosine of the mean vectors of the KNOWN words.

    In the studio this feature was defined in the Arch 1 design, but its graph had no input block for it, so it never reached the
    model (461,825 weights = 900 inputs). It is here only for the optional --similarity-feature flag."""
    def mean(toks):
        vs = [v for v in (glove.vec(t) for t in toks) if v is not None]
        return np.mean(vs, 0) if vs else np.zeros(DIM, dtype=np.float32)

    def cos(a, b):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(a @ b / (na * nb)) if na and nb else 0.0

    out = np.zeros((len(rows), 2), dtype=np.float32)
    for i, r in enumerate(rows):
        o1, o2 = mean(r["o1"]), mean(r["o2"])
        for c, h in enumerate((mean(r["h1"]), mean(r["h2"]))):
            out[i, c] = cos(h, o1) + cos(h, o2)
    return out


# ============================================================================================ models
def mlp(d_in, hidden, dropout):
    """Linear -> ReLU -> Dropout -> Linear(1). The state-dict names (0, 3) are the studio's."""
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


def masked_mean(x, mask):
    """Mean over the real words (padding ignored; unknown words count, with their zero vector). Empty sentence -> zeros."""
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class Arch1(nn.Module):
    """Mean GloVe vector of O1, O2 and h, joined (900 numbers) -> MLP -> one score for h."""

    def __init__(self, matrix, hidden, dropout, n_features=0):
        super().__init__()
        self.register_buffer("emb", torch.as_tensor(matrix), persistent=False)  # frozen, not saved with the weights
        self.classifier = mlp(3 * DIM + n_features, hidden, dropout)

    def score(self, b):
        parts = [masked_mean(self.emb[b[p]], b[p] != 0) for p in ("o1", "o2", "h")]
        if "feature" in b:
            parts.append(b["feature"])
        return self.classifier(torch.cat(parts, -1))[:, 0]


class Arch2(nn.Module):
    """O1 -> h -> O2 as one word sequence, read by a 2-layer GRU; the last state -> MLP -> one score for h."""

    def __init__(self, matrix, hidden, dropout, layers=2):
        super().__init__()
        self.register_buffer("emb", torch.as_tensor(matrix), persistent=False)
        self.reader = nn.GRU(DIM + 3, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.classifier = mlp(hidden, hidden, dropout)

    def story(self, b):
        """Join O1, h, O2: every word gets 3 extra numbers (1 for the sentence it comes from), an all-zero 'marker' word sits
        between the sentences, and the real words are moved to the front (padding to the back). -> x [B, T, 303], mask [B, T]."""
        parts, masks = [], []
        for k, p in enumerate(("o1", "h", "o2")):
            ids = b[p]
            x = self.emb[ids]
            part = torch.zeros(*x.shape[:2], 3, dtype=x.dtype, device=x.device)
            part[..., k] = 1.0
            parts.append(torch.cat([x, part], -1))
            masks.append(ids != 0)
            if k < 2:
                parts.append(torch.zeros(x.shape[0], 1, DIM + 3, dtype=x.dtype, device=x.device))
                masks.append(torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device))
        x, mask = torch.cat(parts, 1), torch.cat(masks, 1)
        order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices  # stable: real items keep their order
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        mask = torch.gather(mask, 1, order)
        n = max(1, int(mask.sum(1).max()))
        return x[:, :n], mask[:, :n]

    def score(self, b):
        x, mask = self.story(b)
        lengths = mask.sum(1).clamp(min=1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  # padding is skipped
        _, h = self.reader(packed)
        return self.classifier(h[-1])[:, 0]  # state of the top layer after the last real word


def logits(model, b, device):
    """[B, 2]: the scores of h1 and h2 for every example (the model ran once over 2B rows)."""
    s = model.score({k: v.to(device) for k, v in b.items() if k != "y"})
    n = s.shape[0] // 2
    return torch.stack([s[:n], s[n:]], 1)


def build_model(cfg, matrix):
    if cfg["arch"] == 1:
        return Arch1(matrix, cfg["hidden"], cfg["dropout"], 1 if cfg["similarity_feature"] else 0)
    return Arch2(matrix, cfg["hidden"], cfg["dropout"])


# ============================================================================================ training
@torch.no_grad()
def evaluate(model, split, device, bs=256):
    """Mean cross-entropy and accuracy (h2 is picked when its probability is > 0.5). Also returns p(h2) per example."""
    model.eval()
    loss, probs = 0.0, []
    for a in range(0, split.n, bs):
        idx = np.arange(a, min(split.n, a + bs))
        b = split.batch(idx)
        lg = logits(model, b, device)
        loss += F.cross_entropy(lg, b["y"].to(device), reduction="sum").item()
        probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p2 = np.concatenate(probs)
    acc = float(((p2 > 0.5).astype(np.int64) == split.labels).mean())
    return {"loss": round(loss / split.n, 5), "acc": round(acc, 5)}, p2


def prepare_splits(cfg, data, names):
    """Tokenize, build the vocabulary over the given splits, and (optionally) the standardised feature."""
    glove = Glove(data)
    rows = {s: load_split(data, s) for s in names}
    vocab = Vocab([rows[s] for s in names], glove)
    feats = {s: None for s in names}
    if cfg.get("similarity_feature"):
        raw = {s: similarity_feature(rows[s], glove) for s in names}
        mu, sd = cfg.get("feature_mean"), cfg.get("feature_std")
        if mu is None:  # standardise with mean / spread of the TRAIN split (both hypotheses)
            mu, sd = float(raw["train"].mean()), float(max(raw["train"].std(), 1e-6))
            cfg["feature_mean"], cfg["feature_std"] = mu, sd
        feats = {s: (raw[s] - mu) / sd for s in names}
    return {s: Split(rows[s], vocab, feats[s]) for s in names}, vocab


def wandb_init(args, cfg):
    """Optional Weights & Biases logging (off unless --wandb). Login via `wandb login` / WANDB_API_KEY, never stored here."""
    if not args.wandb:
        return None
    import wandb
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=pathlib.Path(args.out).name, config=cfg,
                      dir=str(args.out))


def train(args):
    torch.set_num_threads(args.threads)
    cfg = {"arch": args.arch, "seed": args.seed, "similarity_feature": args.similarity_feature,
           **{k: (getattr(args, k) if getattr(args, k) is not None else v) for k, v in DEFAULTS[args.arch].items()},
           "patience": args.patience, "grad_clip": GRAD_CLIP, "optimizer": "adam", "monitor": "val_loss"}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    # Same order of random draws as the studio: seeds -> data -> model weights -> numpy generator for the batch order.
    seed = cfg["seed"]
    torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
    splits, vocab = prepare_splits(cfg, args.data, ("train", "validation"))
    train_s, val_s = splits["train"], splits["validation"]
    print(f"data: {train_s.n} train / {val_s.n} validation examples, vocabulary {len(vocab.ids)} words, "
          f"{len(vocab.unknown)} without a GloVe vector", flush=True)
    model = build_model(cfg, vocab.matrix).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    cfg["params"] = sum(p.numel() for p in params)
    print(f"Arch {cfg['arch']}: {cfg['params']:,} trainable weights; {json.dumps({k: cfg[k] for k in ('lr', 'hidden', 'dropout', 'batch')})}",
          flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    use = np.sort(rng.permutation(train_s.n))  # all training examples (the studio drew this permutation for its "use part" option)
    bs = cfg["batch"]
    steps = (len(use) + bs - 1) // bs
    wb = wandb_init(args, cfg)

    best, bad, epoch, stop = None, 0, 0, "finished all epochs"
    metrics = open(out / "metrics.jsonl", "w")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(use)
        tl, tc, tn = 0.0, 0, 0
        for k in range(steps):
            idx = order[k * bs:(k + 1) * bs]
            b = train_s.batch(idx)
            lg = logits(model, b, device)
            y = b["y"].to(device)
            loss = F.cross_entropy(lg, y)  # softmax over the two scores
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            tl += loss.item() * len(idx)
            tc += int((lg.argmax(-1) == y).sum())
            tn += len(idx)
        tr = {"loss": round(tl / tn, 5), "acc": round(tc / tn, 5)}
        val, p2 = evaluate(model, val_s, device)
        line = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"], "val_loss": val["loss"], "val_acc": val["acc"],
                "minutes": round((time.time() - t0) / 60, 2)}
        metrics.write(json.dumps(line) + "\n")
        metrics.flush()
        print(json.dumps(line), flush=True)
        if wb:
            wb.log({"epoch": epoch, "train/loss": tr["loss"], "train/acc": tr["acc"], "val/loss": val["loss"], "val/acc": val["acc"]}, step=epoch)
        if best is None or val["loss"] < best["val_loss"] - 1e-6:  # keep the weights of the epoch with the lowest validation loss
            best, bad = {"epoch": epoch, "val_loss": val["loss"], "val_acc": val["acc"], "train_loss": tr["loss"], "train_acc": tr["acc"]}, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, out / "best.pt")
            (out / "val_predictions.json").write_text(json.dumps({"epoch": epoch, "p_h2": [round(float(x), 6) for x in p2],
                                                                  "label": val_s.labels.tolist()}))
        else:
            bad += 1
        if time.time() - t0 > args.minutes * 60:
            stop = f"time limit ({args.minutes:g} min)"
            break
        if cfg["patience"] > 0 and bad >= cfg["patience"]:
            stop = f"early stopping: no better validation loss for {bad} epochs"
            break
    metrics.close()
    result = {"best": best, "epochs_run": epoch, "stop_reason": stop, "minutes": round((time.time() - t0) / 60, 2), "params": cfg["params"],
              "device": str(device), "threads": args.threads}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    if wb:
        for k, v in best.items():
            wb.summary[f"best/{k}"] = v
        wb.finish()
    print(json.dumps(result), flush=True)


def load_run(run, data, split):
    cfg = json.loads((pathlib.Path(run) / "config.json").read_text())
    splits, vocab = prepare_splits(cfg, data, (split,))
    ck = torch.load(pathlib.Path(run) / "best.pt", map_location="cpu")
    model = build_model(cfg, vocab.matrix)
    model.load_state_dict(ck["model"])
    return model, splits[split], ck["epoch"]


def evaluate_cmd(args):
    """Best checkpoint on the validation split (safe to repeat)."""
    torch.set_num_threads(args.threads)
    model, val_s, epoch = load_run(args.run, args.data, "validation")
    res, _ = evaluate(model, val_s, torch.device("cpu"))
    print(json.dumps({"split": "validation", "epoch": epoch, **res}))


def test_cmd(args):
    """Best checkpoint on the TEST split, once: a second test is refused (choosing between test results = tuning on test)."""
    torch.set_num_threads(args.threads)
    rp = pathlib.Path(args.run) / "result.json"
    result = json.loads(rp.read_text()) if rp.exists() else {}
    if result.get("test") and not args.force:
        raise SystemExit(f"this run was already tested ({result['test']['at']}): {result['test']}. Use --force only if you know why.")
    model, test_s, epoch = load_run(args.run, args.data, "test")
    res, p2 = evaluate(model, test_s, torch.device("cpu"))
    res.update(epoch=epoch, at=time.strftime("%Y-%m-%dT%H:%M:%S"), forced=bool(args.force), previous=result.get("test"))
    (pathlib.Path(args.run) / "test_predictions.json").write_text(json.dumps({"epoch": epoch, "p_h2": [round(float(x), 6) for x in p2],
                                                                             "label": test_s.labels.tolist()}))
    result["test"] = res
    rp.write_text(json.dumps(result, indent=1))
    print(json.dumps({"split": "test", **res}))


# ============================================================================================ data download
def fetch(url, dest):
    print(f"downloading {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def prepare(args):
    """Download the course dataset (parquet -> jsonl) and GloVe 6B (zip, 822 MB -> 300d as .npy + vocabulary). Skips what exists."""
    data = pathlib.Path(args.data)
    (data / "glove").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out = data / f"{split}.jsonl"
        if out.exists():
            continue
        pqf = data / f"{split}.parquet"
        if not pqf.exists():
            fetch(HF_URL.format(split=split), pqf)
        import pyarrow.parquet as pq  # only needed here
        rows = pq.read_table(pqf).to_pylist()
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} rows", flush=True)
    npy, vocab = data / "glove" / f"6B.{DIM}.npy", data / "glove" / "6B.vocab.txt"
    if not (npy.exists() and vocab.exists()):
        import zipfile
        z = data / "glove" / "glove.6B.zip"
        if not z.exists():
            fetch(GLOVE_URL, z)
        words, rows = [], []
        with zipfile.ZipFile(z) as zf, zf.open(f"glove.6B.{DIM}d.txt") as f:
            for line in f:
                parts = line.decode("utf-8").rstrip("\n").split(" ")
                words.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        np.save(npy, np.stack(rows))
        vocab.write_text("\n".join(words) + "\n")
        print(f"GloVe {DIM}d: {len(words)} words", flush=True)
    print("ok")


# ============================================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="download + convert the dataset and GloVe")
    p.add_argument("--data", required=True)
    p = sub.add_parser("train", help="train one model (defaults = the final configuration of the chosen architecture)")
    p.add_argument("--arch", type=int, choices=(1, 2), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--lr", type=float)
    p.add_argument("--hidden", type=int, help="MLP hidden size (Arch 2: GRU and MLP hidden size)")
    p.add_argument("--dropout", type=float)
    p.add_argument("--batch", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--minutes", type=float, default=MINUTES, help="time limit; the best epoch so far is kept")
    p.add_argument("--similarity-feature", action="store_true",
                   help="Arch 1 only: also feed the standardised cos(h,o1)+cos(h,o2) to the MLP (NOT part of the final model)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (off by default)")
    p.add_argument("--wandb-project", default="art-p1")
    p.add_argument("--wandb-entity", default=None)
    for name, hlp in (("evaluate", "best checkpoint on the validation split"), ("test", "best checkpoint on the test split (once)")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--run", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--threads", type=int, default=2)
        if name == "test":
            p.add_argument("--force", action="store_true", help="test again although the run was already tested")
    args = ap.parse_args()
    if args.cmd == "train" and args.arch == 2 and args.similarity_feature:
        ap.error("--similarity-feature is only implemented for Arch 1")
    {"prepare": prepare, "train": train, "evaluate": evaluate_cmd, "test": test_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
