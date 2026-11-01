"""Server side of the ART (text) source: live previews for the Inputs tab, dataset statistics, feature checks, word lookups,
and the run extras (one-time test, error browser). Used by serve.py; everything here is read-only on the data."""
import functools
import json
import pathlib
import subprocess

import numpy as np

from xcore import store
from xcore import text as T

ROOT = pathlib.Path(__file__).resolve().parents[1]


def art_python():
    """The ART trainer's Python (own venv with wandb; shares torch with the alphaciv venv)."""
    p = ROOT / ".venv-art" / "bin" / "python"
    return str(p) if p.exists() else str(ROOT.parent / "alphaciv" / ".venv" / "bin" / "python")


def _vectors(toks, g, ecfg):
    """What the model gets for one sentence: one vector per token (unknown words per the chosen rule)."""
    unk = T.unknown_vector(g, ecfg["unknown"])
    rows, kept = [], []
    for t in toks:
        v = g.vec(t, ecfg["fallback_lower"])
        if v is None:
            if ecfg["unknown"] == "drop":
                continue
            v = unk
        rows.append(v)
        kept.append(t)
    return (np.stack(rows) if rows else np.zeros((0, g.dim), np.float32)), kept


def _cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return round(float(a @ b / (na * nb)), 4) if na > 0 and nb > 0 else 0.0


def preview(b):
    """One example run through the pipeline: every step's output, known / unknown words, what the model sees, sentence similarities, features."""
    inp = b.get("inputs") or {}
    t = T.text_cfg(inp)
    ecfg = t["embedding"]
    g = T.glove(ecfg["dim"])
    split = b.get("split", "validation")
    rows = T.load_split(split)
    i = max(0, min(int(b.get("index", 0)), len(rows) - 1))
    r = rows[i]
    labels = {s["op"]: T.STEPS[s["op"]]["label"] for s in t["steps"] if s.get("op") in T.STEPS}
    fields, means, toks_all = {}, {}, {}
    for f in T.FIELDS:
        toks, stages = T.run_pipeline(r[f], t["steps"], trace=True)
        words = []
        for w in toks:
            exact = g.index.get(w)
            row = g.row(w, ecfg["fallback_lower"])
            words.append({"w": w, "known": row is not None, "via_lower": exact is None and row is not None, "rank": row})
        V, kept = _vectors(toks, g, ecfg)
        means[f] = V.mean(0) if len(V) else np.zeros(g.dim, np.float32)
        toks_all[f] = toks
        entry = {"raw": r[f], "tokens": words, "stages": [dict(s, label=labels.get(s["op"], s["op"])) for s in stages]}
        if b.get("vectors"):
            k = int(b.get("max_dims", 64))
            entry["vectors"] = [[round(float(x), 3) for x in v[:k]] for v in V[:40]]
            entry["vector_words"] = kept[:40]
            entry["mean"] = [round(float(x), 3) for x in means[f][:k]]
        fields[f] = entry
    sims = {f"{a}-{c}": _cos(means[a], means[c]) for a in T.FIELDS for c in T.FIELDS if a < c}
    feats = []
    env = {"o1": toks_all["o1"], "o2": toks_all["o2"]}
    for k, f in enumerate(inp.get("text_features") or []):
        try:
            fm = T.TextFormula(f.get("formula", ""))
            v1 = fm(dict(env, h=toks_all["h1"], other=toks_all["h2"]), g, ecfg)
            v2 = fm(dict(env, h=toks_all["h2"], other=toks_all["h1"]), g, ecfg)
            feats.append({"index": k, "name": f.get("name"), "h1": round(v1, 4), "h2": round(v2, 4), "off": bool(f.get("off"))})
        except T.FormulaError as e:
            feats.append({"index": k, "name": f.get("name"), "error": str(e)})
    return {"split": split, "index": i, "n": len(rows), "label": r["label"], "fields": fields, "sims": sims, "features": feats, "dim": g.dim,
            "same_words": sorted(set(w.lower() for w in toks_all["h1"]) & set(w.lower() for w in toks_all["h2"])),
            "only_h1": sorted(set(w.lower() for w in toks_all["h1"]) - set(w.lower() for w in toks_all["h2"])),
            "only_h2": sorted(set(w.lower() for w in toks_all["h2"]) - set(w.lower() for w in toks_all["h1"]))}


@functools.lru_cache(maxsize=24)
def _stats(split, skey, ekey):
    steps, ecfg = json.loads(skey), json.loads(ekey)
    g = T.glove(ecfg["dim"])
    rows = T.tokenized(split, steps)
    raw = T.load_split(split)
    n_tok, n_unk, types, unk_types, lens = 0, 0, {}, {}, {f: [] for f in T.FIELDS}
    known_cache = {}
    for r in rows:
        for f in T.FIELDS:
            lens[f].append(len(r[f]))
            for w in r[f]:
                n_tok += 1
                types[w] = types.get(w, 0) + 1
                k = known_cache.get(w)
                if k is None:
                    k = known_cache[w] = g.row(w, ecfg["fallback_lower"]) is not None
                if not k:
                    n_unk += 1
                    unk_types[w] = unk_types.get(w, 0) + 1
    # what every step changes (on the first 2,000 examples, all four sentences)
    impact = []
    on = [s for s in steps if not s.get("off")]
    sample = raw[:2000]
    counts = [{"op": s["op"], "label": T.STEPS[s["op"]]["label"], "changed": 0, "tokens_after": 0} for s in on]
    for r in sample:
        for f in T.FIELDS:
            _, stages = T.run_pipeline(r[f], on, trace=True)
            for c, st in zip(counts, stages):
                c["changed"] += bool(st.get("changed"))
                if isinstance(st.get("value"), list):
                    c["tokens_after"] += len(st["value"])
    for c in counts:
        c["changed_share"] = round(c["changed"] / (4 * len(sample)), 4)
        c["tokens_per_sentence"] = round(c["tokens_after"] / (4 * len(sample)), 2) if c["tokens_after"] else None
        impact.append(c)
    # how different do h1 and h2 look after mean pooling?
    sims, small_diff = [], 0
    for r in rows[:3000]:
        a, b2 = T.mean_vec(r["h1"], g, ecfg), T.mean_vec(r["h2"], g, ecfg)
        sims.append(_cos(a, b2))
        s1, s2 = set(r["h1"]), set(r["h2"])
        small_diff += len(s1 ^ s2) <= 4
    q = lambda xs, p: float(np.percentile(xs, p)) if xs else 0.0  # noqa: E731
    return {"split": split, "examples": len(rows), "tokens": n_tok, "types": len(types), "unknown_tokens": n_unk,
            "unknown_token_share": round(n_unk / max(1, n_tok), 5), "unknown_types": len(unk_types),
            "top_unknown": sorted(unk_types.items(), key=lambda kv: -kv[1])[:30],
            "top_words": sorted(types.items(), key=lambda kv: -kv[1])[:30],
            "lengths": {f: {"mean": round(float(np.mean(v)), 2), "p50": q(v, 50), "p95": q(v, 95), "max": int(max(v) if v else 0),
                            "empty": int(sum(1 for x in v if x == 0))} for f, v in lens.items()},
            "steps": impact, "h1_h2_cosine_mean": round(float(np.mean(sims)), 4), "h1_h2_cosine_p90": round(q(sims, 90), 4),
            "pairs_differing_by_4_words_or_less": round(small_diff / max(1, min(3000, len(rows))), 4)}


def stats(b):
    t = T.text_cfg(b.get("inputs") or {})
    errs = T.check_steps(t["steps"])
    if errs:
        return {"error": "; ".join(errs)}
    return _stats(b.get("split", "train"), T.steps_key(t["steps"]), json.dumps(t["embedding"], sort_keys=True))


@functools.lru_cache(maxsize=64)
def _feature_values(split, skey, ekey, formula):
    steps, ecfg = json.loads(skey), json.loads(ekey)
    return T.feature_values(T.tokenized(split, steps), [{"formula": formula}], T.glove(ecfg["dim"]), ecfg)[:, :, 0]


def feature(b):
    """How much a single own feature says on its own: 'pick the hypothesis with the higher value' as a classifier."""
    inp = b.get("inputs") or {}
    t = T.text_cfg(inp)
    T.TextFormula(b.get("formula", ""))  # raises a readable error
    split = b.get("split", "train")
    v = _feature_values(split, T.steps_key(t["steps"]), json.dumps(t["embedding"], sort_keys=True), b["formula"])
    y = np.array([r["label"] for r in T.load_split(split)])
    good = v[np.arange(len(y)), y]
    bad = v[np.arange(len(y)), 1 - y]
    ties = float(np.mean(np.isclose(good, bad)))
    higher = float(np.mean(good > bad + 1e-9)) + ties / 2
    lower = float(np.mean(good < bad - 1e-9)) + ties / 2
    lo, hi = float(min(good.min(), bad.min())), float(max(good.max(), bad.max()))
    bins = np.linspace(lo, hi if hi > lo else lo + 1, 21)
    hg, _ = np.histogram(good, bins)
    hb, _ = np.histogram(bad, bins)
    diff = good - bad
    helps = [int(i) for i in np.argsort(-diff)[:4]]
    misleads = [int(i) for i in np.argsort(diff)[:4]]
    rows = T.load_split(split)
    ex = lambda i: {"index": i, "h_plausible": rows[i][f"h{y[i] + 1}"], "h_other": rows[i][f"h{2 - y[i]}"], "o1": rows[i]["o1"], "o2": rows[i]["o2"],  # noqa: E731
                    "plausible": round(float(good[i]), 4), "other": round(float(bad[i]), 4)}
    return {"split": split, "n": int(len(y)), "acc_higher": round(higher, 4), "acc_lower": round(lower, 4), "ties": round(ties, 4),
            "mean_plausible": round(float(good.mean()), 4), "mean_other": round(float(bad.mean()), 4),
            "std": round(float(np.concatenate([good, bad]).std()), 4), "bins": [round(float(x), 4) for x in bins],
            "hist_plausible": hg.tolist(), "hist_other": hb.tolist(), "helps": [ex(i) for i in helps], "misleads": [ex(i) for i in misleads]}


def word(b):
    g = T.glove(int(b.get("dim", 300)))
    w = str(b.get("word", "")).strip()
    row = g.row(w, True)
    out = {"word": w, "known": row is not None, "rank": row, "exact": g.index.get(w) is not None}
    if row is not None:
        v = g.vec(w)
        out["vector"] = [round(float(x), 3) for x in v]
        out["norm"] = round(float(np.linalg.norm(v)), 3)
        out["neighbours"] = g.neighbours(w, int(b.get("k", 8)))
    return out


def search(b):
    """Examples whose sentences contain all the given words (raw text, case-insensitive)."""
    split = b.get("split", "validation")
    words = [w.lower() for w in str(b.get("q", "")).split() if w]
    out = []
    for r in T.load_split(split):
        txt = " ".join(r[f] for f in T.FIELDS).lower()
        if all(w in txt for w in words):
            out.append(r["i"])
            if len(out) >= 200:
                break
    return {"split": split, "indices": out}


# ---------------------------------------------------------------------------------------------- runs
def run_test(name, force=False):
    cmd = [art_python(), str(ROOT / "xcore" / "train_supervised.py"), name, "--test"] + (["--force"] if force else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["test failed"])[-1]
        raise ValueError(msg)
    return json.loads(r.stdout.strip().splitlines()[-1])


def run_examples(name, b):
    """Validation (or, after the one test, test) examples with the model's probabilities: wrong ones, unsure ones, ..."""
    d = store.RUNS / name
    split = b.get("split", "validation")
    f = d / ("test_predictions.json" if split == "test" else "val_predictions.json")
    if not f.exists():
        raise KeyError("no predictions yet" if split != "test" else "the run has not been tested")
    pr = json.loads(f.read_text())
    p2, y = np.array(pr["p_h2"]), np.array(pr["label"])
    rows = T.load_split(split)
    p_right = np.where(y == 1, p2, 1 - p2)
    right = (p2 > 0.5).astype(np.int64) == y  # the trainer's rule: a tie counts as h1
    which = b.get("filter", "wrong")
    idx = np.arange(len(y))
    if which == "wrong":
        idx = idx[~right]
        idx = idx[np.argsort(p_right[idx])]          # most confidently wrong first
    elif which == "right":
        idx = idx[right]
        idx = idx[np.argsort(-p_right[idx])]
    elif which == "unsure":
        idx = idx[np.argsort(np.abs(p_right[idx] - 0.5))]
    off, lim = int(b.get("offset", 0)), int(b.get("limit", 20))
    # accuracy by how similar the two hypotheses are (word sets after a plain split)
    def nd(r):
        a, c = set(r["h1"].lower().split()), set(r["h2"].lower().split())
        return len(a ^ c)
    diffs = np.array([nd(r) for r in rows[:len(y)]])
    groups = []
    for lab, lo, hi in (("differ by ≤ 4 words", 0, 4), ("5-10 words", 5, 10), ("more than 10", 11, 999)):
        m = (diffs >= lo) & (diffs <= hi)
        if m.any():
            groups.append({"group": lab, "n": int(m.sum()), "acc": round(float(right[m].mean()), 4)})
    conf = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        c = np.maximum(p2, 1 - p2)
        m = (c >= lo) & (c < lo + 0.1 + (1e-9 if lo == 0.9 else 0))
        if m.any():
            conf.append({"confidence": f"{int(lo * 100)}-{int(lo * 100) + 10} %", "n": int(m.sum()), "acc": round(float(right[m].mean()), 4)})
    return {"split": split, "epoch": pr.get("epoch"), "n": int(len(y)), "acc": round(float(right.mean()), 4), "total": int(len(idx)),
            "groups": groups, "confidence": conf,
            "items": [{"index": int(i), "o1": rows[i]["o1"], "o2": rows[i]["o2"], "h1": rows[i]["h1"], "h2": rows[i]["h2"], "label": int(y[i]),
                       "p_h2": round(float(p2[i]), 4)} for i in idx[off:off + lim]]}
