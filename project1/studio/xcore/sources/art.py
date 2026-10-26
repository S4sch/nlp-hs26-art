"""Source: ART (abductive commonsense reasoning, NLP HS26 project). A dataset, trained with supervised learning in the Studio.

Every example is a tiny story: observation O1 (the beginning), observation O2 (the end) and two hypotheses h1 / h2 about
what happened in between. One hypothesis is plausible, the other not; the model has to pick the plausible one.
Course splits (amarfurt/art-nlp-hs26): train 16,801 / validation 1,000 / test 1,532. Labels are balanced (chance = 50 %).

What the model gets is decided in the Inputs tab (all fixed, nothing learned): the preprocessing pipeline turns every
sentence into words, the frozen GloVe vectors turn every word into numbers, and own text features (formulas) add single
numbers. The ports are word sequences (sets of word vectors in sentence order) plus an optional vector of own features.

I/O formats
- score_each: the model sees (O1, O2, h) and gives ONE score; it runs for h1 and for h2 with the same weights, softmax over
  the two scores. Ports o1, o2, h (+ features of h). Output block: out_score.
- all_in_one: the model sees (O1, O2, h1, h2) at once and gives two class scores. Ports o1, o2, h1, h2 (+ features of
  h1 then h2). Output block: out_class.
"""
import functools
import json

import numpy as np
import torch
import torch.nn as nn

from xcore import graph as G
from xcore import text as T
from xcore.sources import Source
from xcore.sources import art_ops  # noqa: F401  (registers the text blocks)

FORMATS = {
    "score_each": {"label": "Score each hypothesis", "ports": ["o1", "o2", "h"], "output": "out_score",
                   "help": "The model reads (O1, O2, h) and gives one score. It runs twice per example (h = h1, then h = h2) with the same weights; "
                           "a softmax over the two scores gives the probabilities. Both hypotheses are treated exactly alike."},
    "all_in_one": {"label": "All in one", "ports": ["o1", "o2", "h1", "h2"], "output": "out_class",
                   "help": "The model reads O1, O2, h1 and h2 at once and gives two scores (h1 is right / h2 is right). It can compare the hypotheses "
                           "directly, but has to learn that swapping them swaps the answer."},
}
PORT_TEXT = {"o1": ("O1: the beginning", "observation 1, the first sentence of the story"),
             "o2": ("O2: the end", "observation 2, the last sentence of the story"),
             "h": ("h: the hypothesis being scored", "h1 in the first pass, h2 in the second (same weights)"),
             "h1": ("h1: hypothesis 1", "the first candidate for what happened in between"),
             "h2": ("h2: hypothesis 2", "the second candidate")}
FEATURE_IDEAS = [
    {"name": "o2_share", "formula": "share(h, o2)",
     "why": "Plausible hypotheses often mention what the ending is about (the same person, object or place). Share of h's words that appear in O2."},
    {"name": "o2_share_vs_other", "formula": "share(h, o2) - share(other, o2)",
     "why": "The same, compared with the competitor: positive if h shares more with the ending than the other hypothesis does."},
    {"name": "negations", "formula": "neg(h)",
     "why": "Wrong hypotheses were often made by adding a negation to the right one ('He was never lost'). A pure surface cue."},
    {"name": "length", "formula": "len(h)", "why": "Do plausible hypotheses tend to be longer or shorter? A classic annotation artifact to check."},
    {"name": "cos_o2", "formula": "cos(h, o2)", "why": "Do h and the ending talk about similar things (mean word vectors)? Unlike share(), synonyms count too."},
    {"name": "only_new", "formula": "only_in(h, other)", "why": "How many words h has that the other hypothesis lacks: where the two differ."},
]
EMBEDDINGS = [{"name": "glove", "label": "GloVe 6B (Wikipedia + Gigaword 2014, 400k lowercase words)", "dims": [50, 100, 200, 300]}]


def _N(id, op, x, y, **params):
    return {"id": id, "op": op, "params": params, "pos": [x, y]}


def _E(a, b):
    return {"from": a, "to": b}


ARCH1 = {"nodes": [_N("in_o1", "input", 0, 0, port="o1"), _N("in_o2", "input", 0, 160, port="o2"), _N("in_h", "input", 0, 320, port="h"),
                   _N("mean_o1", "set_pool", 280, 0, mode="mean"), _N("mean_o2", "set_pool", 280, 160, mode="mean"), _N("mean_h", "set_pool", 280, 320, mode="mean"),
                   _N("join", "concat", 560, 160), _N("classifier", "mlp", 840, 160, sizes=[256, 1], act="relu", dropout=0.2),
                   _N("score", "out_score", 1120, 160)],
         "edges": [_E("in_o1", "mean_o1.in"), _E("in_o2", "mean_o2.in"), _E("in_h", "mean_h.in"),
                   _E("mean_o1", "join.in"), _E("mean_o2", "join.in"), _E("mean_h", "join.in"), _E("join", "classifier.in"), _E("classifier", "score.score")]}
ARCH1_ALL = {"nodes": [_N("in_o1", "input", 0, 0, port="o1"), _N("in_o2", "input", 0, 150, port="o2"), _N("in_h1", "input", 0, 300, port="h1"),
                       _N("in_h2", "input", 0, 450, port="h2"),
                       _N("mean_o1", "set_pool", 280, 0, mode="mean"), _N("mean_o2", "set_pool", 280, 150, mode="mean"),
                       _N("mean_h1", "set_pool", 280, 300, mode="mean"), _N("mean_h2", "set_pool", 280, 450, mode="mean"),
                       _N("join", "concat", 560, 225), _N("classifier", "mlp", 840, 225, sizes=[256, 2], act="relu", dropout=0.2),
                       _N("answer", "out_class", 1120, 225)],
             "edges": [_E("in_o1", "mean_o1.in"), _E("in_o2", "mean_o2.in"), _E("in_h1", "mean_h1.in"), _E("in_h2", "mean_h2.in"),
                       _E("mean_o1", "join.in"), _E("mean_o2", "join.in"), _E("mean_h1", "join.in"), _E("mean_h2", "join.in"),
                       _E("join", "classifier.in"), _E("classifier", "answer.logits")]}
ARCH1_PAIR = {"nodes": [_N("in_o1", "input", 0, 0, port="o1"), _N("in_o2", "input", 0, 160, port="o2"), _N("in_h", "input", 0, 320, port="h"),
                        _N("pool_o1", "set_pool", 260, 0, mode="mean+max"), _N("pool_o2", "set_pool", 260, 160, mode="mean+max"),
                        _N("pool_h", "set_pool", 260, 320, mode="mean+max"),
                        _N("cmp_o1", "pair_features", 520, 60, a=False, b=True, product=True, difference=True, cosine=True),
                        _N("cmp_o2", "pair_features", 520, 260, a=True, b=False, product=True, difference=True, cosine=True),
                        _N("join", "concat", 780, 160), _N("classifier", "mlp", 1040, 160, sizes=[256, 1], act="relu", dropout=0.3),
                        _N("score", "out_score", 1300, 160)],
              "edges": [_E("in_o1", "pool_o1.in"), _E("in_o2", "pool_o2.in"), _E("in_h", "pool_h.in"),
                        _E("pool_o1", "cmp_o1.a"), _E("pool_h", "cmp_o1.b"), _E("pool_o2", "cmp_o2.a"), _E("pool_h", "cmp_o2.b"),
                        _E("cmp_o1", "join.in"), _E("cmp_o2", "join.in"), _E("join", "classifier.in"), _E("classifier", "score.score")]}


def _arch2(cell, bidir):
    return {"nodes": [_N("in_o1", "input", 0, 0, port="o1"), _N("in_h", "input", 0, 160, port="h"), _N("in_o2", "input", 0, 320, port="o2"),
                      _N("story", "seq_join", 280, 160, marker=True, part_ids=False),
                      _N("reader", "rnn", 560, 160, cell=cell, hidden=128, layers=2, bidirectional=bidir, dropout=0.2),
                      _N("classifier", "mlp", 840, 160, sizes=[128, 1], act="relu", dropout=0.2), _N("score", "out_score", 1120, 160)],
            "edges": [_E("in_o1", "story.in"), _E("in_h", "story.in"), _E("in_o2", "story.in"), _E("story", "reader.in"),
                      _E("reader.last", "classifier.in"), _E("classifier", "score.score")]}


TEMPLATES = [
    {"key": "arch1", "label": "Arch 1: mean of word vectors + 2-layer classifier", "graph": ARCH1, "format": "score_each",
     "description": "The project's architecture 1 as planned: each sentence becomes the mean of its GloVe vectors; [O1, O2, h] (900 numbers) go through "
                    "Linear(900→256) → ReLU → Linear(256→1) = a score for h. Runs for h1 and h2 with the same weights. Only the classifier is trained."},
    {"key": "arch1_all", "label": "Arch 1, all in one (h1 and h2 together)", "graph": ARCH1_ALL, "format": "all_in_one",
     "description": "Same idea, other I/O format: [O1, O2, h1, h2] (1,200 numbers) → Linear → ReLU → Linear(→2 classes). Compare it with the "
                    "'score each' version to see what the format changes."},
    {"key": "arch1_pairs", "label": "Arch 1+: mean+max, comparison features", "graph": ARCH1_PAIR, "format": "score_each",
     "description": "Still only a 2-layer classifier is trained, but its input is richer: mean AND max of the word vectors, and InferSent-style "
                    "comparisons of h with each observation (h·O, |h−O|, cosine). Cite InferSent (Conneau et al. 2017) if you keep it."},
    {"key": "arch2_gru", "label": "Arch 2: GRU reads O1 → h → O2", "graph": _arch2("gru", False), "format": "score_each",
     "description": "The project's architecture 2: the words of O1, h and O2 are joined into one sequence (the story in order) and read by a 2-layer GRU "
                    "(frozen GloVe inputs). Its last state goes through a 2-layer ReLU classifier to the score of h."},
    {"key": "arch2_bilstm", "label": "Arch 2: bidirectional LSTM", "graph": _arch2("lstm", True), "format": "score_each",
     "description": "Like the GRU version, with a 2-layer LSTM that also reads the story backwards. More weights, slower; compare it with the GRU."},
]


def default_inputs(fmt="score_each"):
    return {"text": {"steps": [dict(s) for s in T.DEFAULT_STEPS], "embedding": dict(T.DEFAULT_EMBEDDING), "format": fmt}, "text_features": [],
            "features": [], "extras": {}, "disabled": []}


TRAINING = [
    ("epochs", "Epochs (at most)", "One epoch = every training example once. Early stopping usually ends earlier.", 1, 200, 1, 30),
    ("batch_size", "Batch size", "Examples per update. Small: noisier, more updates per epoch. 32-128 is usual.", 4, 1024, 4, 64),
    ("lr", "Learning rate", "Step size of each update (Adam: 1e-3 is a good start; RNNs often like 5e-4).", 0.00001, 0.1, 0.0001, 0.001),
    ("weight_decay", "Weight decay", "Pulls weights towards 0 (AdamW): a brake against overfitting. 0 = off; try 0.01.", 0, 0.5, 0.001, 0.0),
    ("patience", "Early stopping patience", "Stop when the validation loss has not improved for this many epochs; the best epoch's weights are kept. 0 = never stop early.", 0, 50, 1, 3),
    ("grad_clip", "Gradient clipping", "Largest allowed size of one update's gradient (0 = off). Protects RNNs from 'exploding gradients'.", 0, 100, 0.5, 1.0),
    ("train_fraction", "Use part of the training data", "1 = all 16,801 examples. Lower = faster tries, and shows how much more data would help.", 0.05, 1, 0.05, 1.0),
    ("minutes", "Time limit (minutes)", "Safety net: the run stops after this long and keeps its best epoch.", 1, 600, 1, 60),
    ("seed", "Seed", "Same seed + same design = the same run. Repeat with other seeds to see how much is luck (±2-3 % on 1,000 examples).", 0, 9999, 1, 0),
]
TRAINING_CHOICES = [
    ("optimizer", "Optimizer", "adam: the default. adamw: Adam with proper weight decay. sgd: plain gradient steps (with momentum 0.9), needs a higher lr.",
     ["adam", "adamw", "sgd"], "adam"),
    ("monitor", "Best epoch by", "Which validation number picks the epoch whose weights are kept (and drives early stopping).", ["val_loss", "val_acc"], "val_loss"),
]
TRAINING_BOOLS = [
    ("swap_augment", "Swap augmentation (all in one)", "Each epoch, randomly swap h1 and h2 (and the label) in half of the examples: doubles the "
                                                       "variety and teaches 'swapping swaps the answer'. Only for the 'all in one' format.", False),
    ("wandb", "Log to Weights & Biases", "Mirror the run to W&B (team sascha_ai) with its full config: needed for the project's W&B view.", True),
]


def default_training():
    d = {k: dflt for k, *_, dflt in TRAINING}
    d.update({k: dflt for k, *_, dflt in TRAINING_CHOICES})
    d.update({k: dflt for k, *_, dflt in TRAINING_BOOLS})
    d.update({"wandb_project": "art-p1", "wandb_entity": "sascha_ai", "threads": 2})
    return d


# ---------------------------------------------------------------------------------------------- encoding
def _pad(lists, min_len=1):
    n = max([len(x) for x in lists] + [min_len])
    out = np.zeros((len(lists), n), dtype=np.int64)
    for i, x in enumerate(lists):
        out[i, :len(x)] = x
    return torch.from_numpy(out)


@functools.lru_cache(maxsize=32)
def _feature_cache(split, skey, ekey, formula):
    steps, ecfg = json.loads(skey), json.loads(ekey)
    rows = T.tokenized(split, steps)
    return T.feature_values(rows, [{"formula": formula}], T.glove(ecfg["dim"]), ecfg)[:, :, 0]


def feature_matrix(split, inputs_cfg):
    """[n_rows, 2, F] own features of a split (raw, not standardised)."""
    t = T.text_cfg(inputs_cfg)
    feats = T.feature_list(inputs_cfg)
    if not feats:
        return np.zeros((len(T.load_split(split)), 2, 0), dtype=np.float32)
    sk, ek = T.steps_key(t["steps"]), json.dumps(t["embedding"], sort_keys=True)
    return np.stack([_feature_cache(split, sk, ek, f["formula"]) for f in feats], -1)


def feature_scaling(inputs_cfg):
    """Mean / spread of every own feature on the training split (features are standardised with these)."""
    f = feature_matrix("train", inputs_cfg).reshape(-1, len(T.feature_list(inputs_cfg)))
    if f.shape[1] == 0:
        return np.zeros(0, np.float32), np.ones(0, np.float32)
    return f.mean(0), np.maximum(f.std(0), 1e-6)


class Encoded:
    """Splits turned into word ids (one shared vocabulary), the frozen vector table and standardised own features."""

    def __init__(self, inputs_cfg, splits=("train", "validation"), indices=None):
        t = T.text_cfg(inputs_cfg)
        self.fmt, self.ecfg, self.steps = t["format"], t["embedding"], t["steps"]
        self.g = T.glove(self.ecfg["dim"])
        self.splits = splits
        self.tok, self.rows = {}, {}
        for s in splits:
            rows, toks = T.load_split(s), T.tokenized(s, self.steps)
            idx = indices.get(s) if isinstance(indices, dict) and s in indices else range(len(rows))
            self.rows[s] = [rows[i] for i in idx]
            self.tok[s] = [toks[i] for i in idx]
        self.vocab, self.matrix, self.unknown = T.build_vocab((r[f] for s in splits for r in self.tok[s] for f in T.FIELDS), self.g, self.ecfg)
        self.ids = {s: {f: [T.encode(r[f], self.vocab, self.ecfg) for r in self.tok[s]] for f in T.FIELDS} for s in splits}
        self.labels = {s: np.array([r["label"] for r in self.rows[s]], dtype=np.int64) for s in splits}
        nf = len(T.feature_list(inputs_cfg))
        self.n_features = nf
        self.feat = {}
        if nf:
            mu, sd = feature_scaling(inputs_cfg)
            self.scaling = {"mean": mu.tolist(), "std": sd.tolist()}
            for s in splits:
                full = feature_matrix(s, inputs_cfg)
                sel = [r["i"] for r in self.rows[s]]
                self.feat[s] = ((full[sel] - mu) / sd).astype(np.float32)

    def n(self, split):
        return len(self.rows[split])

    def batch(self, split, idx, swap=None):
        """Tensors for examples idx. score_each: 2 * len(idx) rows (all h1 rows, then all h2 rows). swap: bool array (all in one only)."""
        I = self.ids[split]
        y = torch.from_numpy(self.labels[split][idx].copy())
        F = self.feat.get(split)
        if self.fmt == "score_each":
            o1 = [I["o1"][i] for i in idx] * 2
            o2 = [I["o2"][i] for i in idx] * 2
            h = [I["h1"][i] for i in idx] + [I["h2"][i] for i in idx]
            ids = {"o1": _pad(o1), "o2": _pad(o2), "h": _pad(h)}
            feats = torch.from_numpy(np.concatenate([F[idx, 0], F[idx, 1]], 0)) if F is not None else None
        else:
            h1 = [I["h1"][i] for i in idx]
            h2 = [I["h2"][i] for i in idx]
            fe = F[idx] if F is not None else None
            if swap is not None and swap.any():
                for k in np.flatnonzero(swap):
                    h1[k], h2[k] = h2[k], h1[k]
                y = torch.where(torch.from_numpy(swap), 1 - y, y)
                if fe is not None:
                    fe = fe.copy()
                    fe[swap] = fe[swap][:, ::-1]
            ids = {"o1": _pad([I["o1"][i] for i in idx]), "o2": _pad([I["o2"][i] for i in idx]), "h1": _pad(h1), "h2": _pad(h2)}
            feats = torch.from_numpy(fe.reshape(len(idx), -1).copy()) if fe is not None else None
        b = {"ids": ids, "y": y}
        if feats is not None:
            b["features"] = feats
        return b


class ArtModel(nn.Module):
    """The design's graph plus the frozen word-vector table in front of it (the table is not saved with the weights)."""

    def __init__(self, src, design, matrix=None):
        super().__init__()
        inp = design.get("inputs", {})
        self.fmt = T.text_cfg(inp)["format"]
        self.net = G.GraphNet(design["graph"], src.input_metas(inp))
        dim = T.emb_cfg(inp)["dim"]
        self.register_buffer("emb", torch.as_tensor(matrix) if matrix is not None else torch.zeros(2, dim), persistent=False)

    def inputs(self, batch):
        if "vectors" in batch:  # previews: already looked up
            return batch["vectors"]
        out = {p: G.SetVal(self.emb[ids], ids != 0, None) for p, ids in batch["ids"].items()}
        if "features" in batch:
            out["features"] = batch["features"]
        return out

    def logits(self, batch):
        """[B, 2]: the scores of h1 and h2 for every example."""
        out = self.net.run(self.inputs(batch))["out"]
        if self.fmt == "score_each":
            if "score" not in out:
                raise ValueError("the graph needs an 'Output: score of this hypothesis' block (format: score each hypothesis)")
            s = out["score"]
            n = s.shape[0] // 2
            return torch.stack([s[:n], s[n:]], 1)
        if "logits" not in out:
            raise ValueError("the graph needs an 'Output: which hypothesis' block (format: all in one)")
        return out["logits"]


# ---------------------------------------------------------------------------------------------- the source
class ArtSource(Source):
    key = "art"
    label = "ART: the more plausible explanation"
    tagline = "NLP HS26 project: O1, O2 and two hypotheses; pick the plausible one. GloVe + your preprocessing; supervised training in the Studio"
    description = ("Abductive commonsense reasoning (αNLI): each example is a short story with a beginning (O1), an end (O2) and two hypotheses "
                   "about what happened in between. The model must pick the plausible one. The dataset was filtered to pairs that BERT got wrong, "
                   "and the wrong hypothesis was often made by changing a few words of the right one, so surface cues help only a little. "
                   "Train 16,801 / validation 1,000 / test 1,532 (amarfurt/art-nlp-hs26); balanced labels, chance = 50 %.")
    geometry = "none"
    trainer = "supervised"
    kind = "dataset"
    icon = "book"

    def input_metas(self, inputs_cfg, scenario=None):
        t = T.text_cfg(inputs_cfg or {})
        d = int(t["embedding"]["dim"])
        fmt = t["format"] if t["format"] in FORMATS else "score_each"
        metas = {p: G.Meta("set", d, name=p) for p in FORMATS[fmt]["ports"]}
        nf = len(T.feature_list(inputs_cfg or {}))
        if nf:
            metas["features"] = G.Meta("vec", nf if fmt == "score_each" else 2 * nf)
        return metas

    def ctx_sizes(self, scenario=None):
        return {"o1": 10, "o2": 10, "h": 10, "h1": 10, "h2": 10, "o1+h+o2": 32, "o1+h1+h2+o2": 42}

    def map_channels(self, inputs_cfg):
        return []

    def map_port(self):
        return None

    def default_inputs(self, template=None):
        t = next((x for x in TEMPLATES if x["key"] == template), None)
        return default_inputs(t["format"] if t else "score_each")

    def default_training(self):
        return default_training()

    def analyse_inputs(self, inputs_cfg):
        """Text-specific problems: pipeline order, formulas, embedding size."""
        t = T.text_cfg(inputs_cfg)
        errs = [{"msg": m} for m in T.check_steps(t["steps"])]
        if int(t["embedding"]["dim"]) not in (50, 100, 200, 300):
            errs.append({"msg": "GloVe 6B comes in 50, 100, 200 or 300 dimensions"})
        if t["format"] not in FORMATS:
            errs.append({"msg": f"unknown I/O format {t['format']}"})
        return {"pipeline_errors": errs, "feature_errors": T.check_features(inputs_cfg)}

    def output_check(self, spec, inputs_cfg):
        fmt = T.text_cfg(inputs_cfg)["format"]
        want = FORMATS.get(fmt, FORMATS["score_each"])["output"]
        have = [n for n in spec.get("nodes", []) if n["op"] in ("out_score", "out_class")]
        errs = []
        if not have:
            errs.append({"node": None, "msg": f"no output block: add '{G.OPS[want].label}' (the format is '{FORMATS[fmt]['label']}')"})
        for n in have:
            if n["op"] != want:
                errs.append({"node": n["id"], "msg": f"the I/O format '{FORMATS[fmt]['label']}' needs '{G.OPS[want].label}' here (change the block or the format in Inputs)"})
        if len(have) > 1:
            errs.append({"node": have[1]["id"], "msg": "only one output block"})
        return errs

    def reward_events(self):
        return []

    def info(self):
        return {
            "key": self.key, "label": self.label, "tagline": self.tagline, "description": self.description, "geometry": self.geometry,
            "trainer": self.trainer, "kind": self.kind, "icon": self.icon,
            "ports": [{"port": p, "kind": "set", "title": PORT_TEXT[p][0], "size": "words × word-vector size",
                       "description": PORT_TEXT[p][1] + ". One vector per word, in sentence order (a set with positions 1..n)."} for p in ("o1", "o2", "h", "h1", "h2")]
                     + [{"port": "features", "kind": "vec", "title": "Own text features", "size": "one number per formula (per hypothesis)",
                         "description": "Numbers you define in Inputs (e.g. word overlap of h with O2), standardised on the training split."}],
            "decision": "Pick the plausible hypothesis: a score for h1 and h2 (softmax), or two class scores. Measured by accuracy.",
            "outputs": ["out_score (score each hypothesis)", "out_class (all in one: 2 classes)"],
            "actions": ["h1", "h2"],
            "formats": FORMATS, "steps": T.STEPS, "default_steps": T.DEFAULT_STEPS, "embeddings": EMBEDDINGS, "unknown": T.UNKNOWN,
            "functions": [{"name": k, "signature": s, "help": h} for k, (s, h) in T.FUNCS.items()], "text_vars": T.TEXT_VARS,
            "feature_ideas": FEATURE_IDEAS, "negations": T.NEGATIONS,
            "reward": {"events": []},
            "training": {"numbers": [{"key": k, "label": l, "help": h, "min": mn, "max": mx, "step": st, "default": d} for k, l, h, mn, mx, st, d in TRAINING],
                         "choices": [{"key": k, "label": l, "help": h, "choices": c, "default": d} for k, l, h, c, d in TRAINING_CHOICES],
                         "bools": [{"key": k, "label": l, "help": h, "default": d} for k, l, h, d in TRAINING_BOOLS]},
            "metrics": {"val/acc": "share of validation examples where the plausible hypothesis got the higher score",
                        "val/loss": "cross-entropy on validation (lower = more confident and right)",
                        "train/acc": "accuracy on the training batches of the epoch (with dropout on)", "train/loss": "cross-entropy on training"},
        }

    def templates(self):
        return TEMPLATES

    # ---- real examples for Look inside / Flow (validation split)
    def sample(self, n=8, seed=0, inputs_cfg=None, scenario=None, **kw):
        inputs_cfg = inputs_cfg or {}
        fmt = T.text_cfg(inputs_cfg)["format"]
        rows = T.load_split("validation")
        rng = np.random.default_rng(1000 + seed)
        n_ex = n // 2 if fmt == "score_each" else n
        picks = sorted(rng.choice(len(rows), size=n_ex, replace=False).tolist())
        enc = Encoded(inputs_cfg, splits=("validation",), indices={"validation": picks})
        b = enc.batch("validation", np.arange(n_ex))
        emb = torch.as_tensor(enc.matrix)
        order = []  # preview row k -> batch row
        if fmt == "score_each":
            for e in range(n_ex):
                order += [e, n_ex + e]  # h1, h2 of the same example next to each other
        else:
            order = list(range(n_ex))
        order_t = torch.tensor(order)
        vectors = {}
        for p, ids in b["ids"].items():
            ids = ids[order_t]
            vectors[p] = G.SetVal(emb[ids], ids != 0, None)
        if "features" in b:
            vectors["features"] = b["features"][order_t]
        inv = {i: w for w, i in enc.vocab.items()}
        positions, words = [], []
        for k, r in enumerate(order):
            e = r % n_ex
            row = enc.rows["validation"][e]
            cand = (1 if r >= n_ex else 0) if fmt == "score_each" else None
            pos = {"example": f"validation #{row['i']}"}
            if cand is not None:
                pos["scoring"] = f"h{cand + 1} ({'plausible' if cand == row['label'] else 'not plausible'})"
            else:
                pos["plausible"] = f"h{row['label'] + 1}"
            positions.append(pos)
            words.append({p: [inv.get(int(j), "<unk>") for j in b["ids"][p][r].tolist() if j != 0] for p in b["ids"]})
        batch = {"vectors": vectors, "words": words, "rows": [enc.rows["validation"][r % n_ex] for r in order], "fmt": fmt,
                 "candidates": [(1 if r >= n_ex else 0) if fmt == "score_each" else None for r in order]}
        return {"batch": batch, "positions": positions}

    def preview_base(self, batch, i):
        r = batch["rows"][i]
        return {"kind": "text", "fmt": batch["fmt"], "candidate": batch["candidates"][i], "label": r["label"],
                "text": {f: r[f] for f in T.FIELDS}, "words": batch["words"][i]}


SOURCE = ArtSource()
