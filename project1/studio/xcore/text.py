"""Text inputs for dataset sources (ART): the preprocessing pipeline, the frozen word vectors and own text features.

Everything here is *fixed* (nothing is learned): it decides which numbers the model gets. The design keeps it in
`inputs.text` (pipeline + embedding + I/O format) and `inputs.text_features` (own numbers from formulas):

    "text": {"steps": [{"op": "normalize"}, {"op": "lowercase"}, {"op": "tokenize", "mode": "glove"}, {"op": "remove_stopwords", "off": true}],
             "embedding": {"name": "glove", "dim": 300, "unknown": "zero", "fallback_lower": true},
             "format": "score_each"}
    "text_features": [{"name": "overlap_o2", "formula": "share(h, o2)"}]

Pipeline: steps run top to bottom. Steps above "tokenize" work on the raw sentence (a string), steps below it on the list
of words. Each step can be switched off ("off": true) to see what it changes.
"""
import ast
import functools
import json
import math
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets" / "art"
SPLITS = ("train", "validation", "test")
FIELDS = ("o1", "o2", "h1", "h2")
RAW = {"o1": "observation_1", "o2": "observation_2", "h1": "hypothesis_1", "h2": "hypothesis_2"}

STOPWORDS = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself
they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
shouldn wasn weren won wouldn n't 's 're 've 'll 'd 'm""".split())
NEGATIONS = ["not", "n't", "no", "never", "nothing", "nobody", "none", "nowhere", "neither", "nor", "without", "cannot"]
KEEP_DEFAULT = NEGATIONS + ["before", "after", "then", "because", "but", "so", "again", "too", "only", "until", "while"]
PUNCT_RE = re.compile(r"^[^\w]+$")
CONTRACTIONS_TEXT = [(r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"\bshan't\b", "shall not"), (r"n't\b", " not"), (r"'m\b", " am"),
                     (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would")]
CONTRACTIONS_TOK = {"n't": "not", "'m": "am", "'re": "are", "'ve": "have", "'ll": "will", "'d": "would", "ca": "can", "wo": "will", "sha": "shall"}


# ---------------------------------------------------------------------------------------------- tokenizers
def tok_glove(s):
    """Like the Stanford tokenizer GloVe was trained with: punctuation split off, "didn't" -> "did" + "n't", "she's" -> "she" + "'s"."""
    s = re.sub(r"(?i)([a-z])n't\b", r"\1 n't", s)
    s = re.sub(r"(?i)(?<=[a-z])'(s|m|d|ll|re|ve)\b", r" '\1", s)
    return re.findall(r"n't|'(?:s|m|d|ll|re|ve)\b|[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\sA-Za-z0-9]", s, flags=re.I)


def tok_whitespace(s):
    """Split at spaces only: punctuation stays glued to the word ("home." is one token)."""
    return s.split()


def tok_letters(s):
    """Only runs of letters / digits / apostrophes; all punctuation disappears."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", s)


TOKENIZERS = {"glove": tok_glove, "whitespace": tok_whitespace, "letters": tok_letters}


# ---------------------------------------------------------------------------------------------- pipeline steps
def _words(v):
    if isinstance(v, (list, tuple)):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    return [w.strip().lower() for w in str(v or "").replace(",", " ").split() if w.strip()]


def _pairs(v):
    out = {}
    for part in str(v or "").split(","):
        a, sep, b = part.partition("=")
        if sep and a.strip():
            out[a.strip().lower()] = b.strip()
    return out


STEPS = {
    "normalize": {"label": "Normalize characters", "where": "text",
                  "help": "Curly quotes ’ “ ” become ' and \", dashes become -, repeated spaces one space.",
                  "why": "\"didn’t\" with a curly apostrophe is a different string from \"didn't\": the tokenizer would not split it and GloVe would not know it.",
                  "params": []},
    "lowercase": {"label": "Lowercase", "where": "both",
                  "help": "Every letter becomes lowercase.",
                  "why": "GloVe 6B only has lowercase words. Without it, \"The\" or \"Tom\" have no vector (unless 'try lowercase' in the embedding settings catches them).",
                  "params": []},
    "expand_contractions": {"label": "Expand contractions", "where": "both",
                            "help": "didn't -> did not, I'm -> I am, she'll -> she will ('s stays: it can be 'is' or a possessive).",
                            "why": "Makes the negation an ordinary word (\"not\"). GloVe also knows \"n't\", so this mainly matters for stopword removal and own features.",
                            "params": []},
    "tokenize": {"label": "Tokenize (split into words)", "where": "split",
                 "help": "Turns the sentence into a list of words (tokens). Steps above work on the sentence, steps below on the word list.",
                 "why": "The vectors are looked up per token, so the split must match how GloVe's text was split: 'glove' does that.",
                 "params": [{"key": "mode", "type": "choice", "default": "glove", "choices": list(TOKENIZERS),
                             "help": "glove: like the Stanford tokenizer (punctuation separate, did + n't). whitespace: split at spaces only "
                                     "(\"home.\" stays one token, mostly unknown). letters: words only, punctuation dropped."}]},
    "remove_punct": {"label": "Remove punctuation", "where": "tokens",
                     "help": "Drops tokens that are only punctuation (. , ! ? ' ...).",
                     "why": "A full stop in every sentence adds the same vector to every mean: it shrinks the differences between sentences a little.",
                     "params": []},
    "remove_stopwords": {"label": "Remove stopwords", "where": "tokens",
                         "help": "Drops very common words (the, a, of, is, ...), except the ones in 'keep'.",
                         "why": "Stopwords dominate a mean of word vectors. But some carry the meaning here: not, never, before, after. Keep those.",
                         "params": [{"key": "keep", "type": "words", "default": " ".join(KEEP_DEFAULT),
                                     "help": "Words that stay even though they are stopwords."}]},
    "drop_words": {"label": "Drop these words", "where": "tokens", "help": "Removes every word in the list.",
                   "why": "Test what one word (or a group) contributes.", "params": [{"key": "words", "type": "words", "default": "", "help": "space separated"}]},
    "replace_words": {"label": "Replace words", "where": "tokens", "help": "a=b, c=d: every 'a' becomes 'b'.",
                      "why": "Map words GloVe does not know to ones it does, or merge words (e.g. names -> 'he').",
                      "params": [{"key": "pairs", "type": "text", "default": "", "help": "word=replacement, comma separated"}]},
    "max_len": {"label": "Cut to a maximum length", "where": "tokens", "help": "Keeps only the first N tokens of each sentence.",
                "why": "Sentences here are short (about 9 words), so this rarely bites; for RNNs it caps the cost.",
                "params": [{"key": "n", "type": "int", "default": 40, "min": 1, "max": 200, "help": "maximum number of tokens"}]},
}
DEFAULT_STEPS = [{"op": "normalize"}, {"op": "lowercase"}, {"op": "tokenize", "mode": "glove"}, {"op": "remove_punct", "off": True},
                 {"op": "remove_stopwords", "off": True, "keep": " ".join(KEEP_DEFAULT)}]
DEFAULT_EMBEDDING = {"name": "glove", "dim": 300, "unknown": "zero", "fallback_lower": True}
UNKNOWN = {"zero": "a vector of zeros (counts as a word in a mean: it pulls the mean towards 0)",
           "drop": "left out (as if the word was not there)",
           "mean": "the average of all GloVe vectors (a 'typical word')",
           "random": "one fixed random vector for every unknown word (all unknown words look the same)"}


def step_param(st, key):
    spec = next((p for p in STEPS[st["op"]]["params"] if p["key"] == key), None)
    return st.get(key, spec["default"] if spec else None)


def check_steps(steps):
    """Problems with a pipeline (empty list = fine)."""
    errs = []
    on = [s for s in steps if not s.get("off")]
    toks = [i for i, s in enumerate(on) if s.get("op") == "tokenize"]
    if len(toks) != 1:
        errs.append("the pipeline needs exactly one (switched on) 'Tokenize' step")
    for i, s in enumerate(on):
        if s.get("op") not in STEPS:
            errs.append(f"unknown step '{s.get('op')}'")
            continue
        where = STEPS[s["op"]]["where"]
        if toks and where == "tokens" and i < toks[0]:
            errs.append(f"'{STEPS[s['op']]['label']}' works on words: move it below 'Tokenize'")
        if toks and where == "text" and i > toks[0]:
            errs.append(f"'{STEPS[s['op']]['label']}' works on the raw sentence: move it above 'Tokenize'")
    return errs


def _text_step(op, st, s):
    if op == "normalize":
        s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
        return re.sub(r"\s+", " ", s).strip()
    if op == "lowercase":
        return s.lower()
    if op == "expand_contractions":
        for a, b in CONTRACTIONS_TEXT:
            s = re.sub(a, b, s, flags=re.I)
        return s
    return s


def _token_step(op, st, toks):
    if op == "lowercase":
        return [t.lower() for t in toks]
    if op == "remove_punct":
        return [t for t in toks if not PUNCT_RE.match(t)]
    if op == "remove_stopwords":
        keep = set(_words(step_param(st, "keep")))
        return [t for t in toks if t.lower() not in STOPWORDS or t.lower() in keep]
    if op == "drop_words":
        drop = set(_words(step_param(st, "words")))
        return [t for t in toks if t.lower() not in drop]
    if op == "replace_words":
        mp = _pairs(step_param(st, "pairs"))
        out = []
        for t in toks:
            r = mp.get(t.lower())
            out.extend(r.split() if r is not None else [t])
        return out
    if op == "max_len":
        return toks[:max(1, int(step_param(st, "n")))]
    return toks


def run_pipeline(text, steps, trace=False):
    """Apply the steps to one sentence. Returns the tokens (and, with trace, what every step produced)."""
    val, stages = str(text or ""), []
    tokenized = False
    for st in steps:
        op = st.get("op")
        if st.get("off") or op not in STEPS:
            if trace:
                stages.append({"op": op, "off": True})
            continue
        before = val
        if op == "tokenize":
            val = TOKENIZERS.get(step_param(st, "mode"), tok_glove)(val)
            tokenized = True
        elif tokenized:
            if op == "expand_contractions":  # "did n't" -> "did not" (also "ca n't" -> "can not")
                out = []
                for i, t in enumerate(val):
                    lo = t.lower()
                    nxt = val[i + 1].lower() if i + 1 < len(val) else ""
                    if lo in ("ca", "wo", "sha") and nxt == "n't":
                        out.append(CONTRACTIONS_TOK[lo])
                    elif lo in CONTRACTIONS_TOK and lo not in ("ca", "wo", "sha"):
                        out.append(CONTRACTIONS_TOK[lo])
                    else:
                        out.append(t)
                val = out
            else:
                val = _token_step(op, st, val)
        else:
            val = _text_step(op, st, val)
        if trace:
            stages.append({"op": op, "value": val if isinstance(val, str) else list(val), "changed": before != val})
    if not tokenized:
        val = tok_glove(val)
    return (val, stages) if trace else val


# ---------------------------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=None)
def load_split(split):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    rows = []
    with open(DATA / f"{split}.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append({"i": i, "o1": r["observation_1"], "o2": r["observation_2"], "h1": r["hypothesis_1"], "h2": r["hypothesis_2"],
                         "label": int(r["label"]) - 1})  # 0 = h1 is the plausible one, 1 = h2
    return rows


def steps_key(steps):
    return json.dumps([s for s in steps if not s.get("off")], sort_keys=True)


@functools.lru_cache(maxsize=16)
def _tokenized(split, skey):
    steps = json.loads(skey)
    return [{f: run_pipeline(r[f], steps) for f in FIELDS} for r in load_split(split)]


def tokenized(split, steps):
    """Every row of a split, run through the pipeline (cached per pipeline)."""
    return _tokenized(split, steps_key(steps))


# ---------------------------------------------------------------------------------------------- word vectors
class Glove:
    """GloVe 6B (Wikipedia + Gigaword, 400k lowercase words), memory-mapped: only looked-up rows are read from disk."""

    def __init__(self, dim):
        if dim not in (50, 100, 200, 300):
            raise ValueError("GloVe 6B comes in 50, 100, 200 or 300 dimensions")
        self.dim = dim
        self.M = np.load(DATA / "glove" / f"6B.{dim}.npy", mmap_mode="r")
        self.index = vocab_index()
        self._mean = None

    def row(self, w, fallback_lower=True):
        r = self.index.get(w)
        if r is None and fallback_lower:
            r = self.index.get(w.lower())
        return r

    def vec(self, w, fallback_lower=True):
        r = self.row(w, fallback_lower)
        return None if r is None else np.asarray(self.M[r], dtype=np.float32)

    def mean_vector(self):
        if self._mean is None:
            p = DATA / "glove" / f"6B.{self.dim}.mean.npy"
            if p.exists():
                self._mean = np.load(p)
            else:
                acc = np.zeros(self.dim, dtype=np.float64)
                for a in range(0, self.M.shape[0], 50000):
                    acc += np.asarray(self.M[a:a + 50000], dtype=np.float64).sum(0)
                self._mean = (acc / self.M.shape[0]).astype(np.float32)
                try:
                    np.save(p, self._mean)
                except OSError:
                    pass
        return self._mean

    def neighbours(self, w, k=8, top=50000):
        v = self.vec(w)
        if v is None:
            return []
        M = np.asarray(self.M[:top], dtype=np.float32)
        sims = M @ v / (np.linalg.norm(M, axis=1) * (np.linalg.norm(v) or 1) + 1e-8)
        order = np.argsort(-sims)[:k + 1]
        words = vocab_list()
        return [{"word": words[j], "sim": round(float(sims[j]), 3)} for j in order if words[j] != w.lower()][:k]


@functools.lru_cache(maxsize=1)
def vocab_list():
    return (DATA / "glove" / "6B.vocab.txt").read_text().split("\n")[:-1]


@functools.lru_cache(maxsize=1)
def vocab_index():
    return {w: i for i, w in enumerate(vocab_list())}


@functools.lru_cache(maxsize=4)
def glove(dim):
    return Glove(int(dim))


def emb_cfg(inputs):
    return dict(DEFAULT_EMBEDDING, **((inputs.get("text") or {}).get("embedding") or {}))


def text_cfg(inputs):
    t = dict(inputs.get("text") or {})
    t.setdefault("steps", [dict(s) for s in DEFAULT_STEPS])
    t["embedding"] = emb_cfg(inputs)
    t.setdefault("format", "score_each")
    return t


def unknown_vector(g, how, seed=1234):
    if how == "mean":
        return g.mean_vector()
    if how == "random":
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(g.dim) * 0.4).astype(np.float32)
    return np.zeros(g.dim, dtype=np.float32)


def build_vocab(token_lists, g, ecfg):
    """Vocabulary of a run: row 0 = padding, row 1 = unknown words. Returns (word -> id, matrix [V, dim], unknown words)."""
    words, unknown = {}, set()
    for toks in token_lists:
        for t in toks:
            if t in words or t in unknown:
                continue
            if g.row(t, ecfg["fallback_lower"]) is None:
                unknown.add(t)
            else:
                words[t] = None
    ids = {w: i + 2 for i, w in enumerate(sorted(words))}
    M = np.zeros((len(ids) + 2, g.dim), dtype=np.float32)
    M[1] = unknown_vector(g, ecfg["unknown"])
    for w, i in ids.items():
        M[i] = g.vec(w, ecfg["fallback_lower"])
    return ids, M, unknown


def encode(toks, ids, ecfg):
    out = []
    for t in toks:
        j = ids.get(t)
        if j is None:
            if ecfg["unknown"] == "drop":
                continue
            j = 1
        out.append(j)
    return out


# ---------------------------------------------------------------------------------------------- own text features (formulas)
FUNCS = {
    "len": ("len(x)", "number of words in x"),
    "overlap": ("overlap(a, b)", "how many different words of a also appear in b"),
    "share": ("share(a, b)", "overlap(a, b) divided by the number of different words in a (0..1)"),
    "jaccard": ("jaccard(a, b)", "shared words / all words of a and b together (0..1)"),
    "only_in": ("only_in(a, b)", "how many different words of a do NOT appear in b"),
    "cos": ("cos(a, b)", "cosine similarity of the mean word vectors of a and b (-1..1): do they talk about similar things?"),
    "count": ("count(x, 'w1', 'w2', ...)", "how often the listed words occur in x"),
    "has": ("has(x, 'w1', ...)", "1 if any of the words occurs in x, else 0"),
    "neg": ("neg(x)", f"number of negation words in x ({', '.join(NEGATIONS)})"),
    "unk": ("unk(x)", "number of words in x that have no word vector"),
    "min": ("min(a, b)", "smaller of two numbers"), "max": ("max(a, b)", "larger of two numbers"), "abs": ("abs(x)", "absolute value"),
    "log": ("log(x)", "log(1 + x) (squashes big counts)"),
}
TEXT_VARS = {"o1": "observation 1 (the beginning)", "o2": "observation 2 (the end)", "h": "the hypothesis being scored",
             "other": "the other hypothesis (the competitor)"}


class FormulaError(ValueError):
    pass


class TextFormula:
    """Safe little expression language over the token lists of one example (see FUNCS / TEXT_VARS)."""

    def __init__(self, src):
        self.src = src
        try:
            self.tree = ast.parse(str(src or "").strip(), mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"syntax: {e.msg}")
        self.uses_vectors = False
        self._check(self.tree.body)

    def _check(self, n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._check(n.left), self._check(n.right)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            self._check(n.operand)
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)):
            self._check(n.left), self._check(n.comparators[0])
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
            pass
        elif isinstance(n, ast.Name):
            if n.id not in TEXT_VARS:
                raise FormulaError(f"unknown name '{n.id}' (texts: {', '.join(TEXT_VARS)}; words go in quotes: 'not')")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id not in FUNCS:
                raise FormulaError(f"unknown function '{n.func.id}'")
            if n.keywords:
                raise FormulaError("no name=value arguments here")
            if n.func.id == "cos":
                self.uses_vectors = True
            for a in n.args:
                self._check(a)
        else:
            raise FormulaError(f"not allowed here: {ast.dump(n)[:40]}")

    def __call__(self, env, g=None, ecfg=None):
        return float(self._ev(self.tree.body, env, g, ecfg))

    def _text(self, v):
        if not isinstance(v, list):
            raise FormulaError("this function needs a text (o1, o2, h, other)")
        return v

    def _ev(self, n, env, g, ecfg):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.UnaryOp):
            v = self._ev(n.operand, env, g, ecfg)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = self._ev(n.left, env, g, ecfg), self._ev(n.right, env, g, ecfg)
            if isinstance(a, (list, str)) or isinstance(b, (list, str)):
                raise FormulaError("arithmetic needs numbers: use len(x), overlap(a, b) ... on texts")
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            return a / b if b else 0.0
        if isinstance(n, ast.Compare):
            a, b = self._ev(n.left, env, g, ecfg), self._ev(n.comparators[0], env, g, ecfg)
            op = n.ops[0]
            return float({ast.Gt: a > b, ast.Lt: a < b, ast.GtE: a >= b, ast.LtE: a <= b, ast.Eq: a == b, ast.NotEq: a != b}[type(op)])
        f = n.func.id
        args = [self._ev(a, env, g, ecfg) for a in n.args]
        need = {"len": 1, "overlap": 2, "share": 2, "jaccard": 2, "only_in": 2, "cos": 2, "neg": 1, "unk": 1, "min": 2, "max": 2, "abs": 1, "log": 1}
        if f in need and len(args) != need[f]:
            raise FormulaError(f"{f}() takes {need[f]} argument{'s' if need[f] > 1 else ''}")
        low = lambda t: {w.lower() for w in self._text(t)}  # noqa: E731
        if f == "len":
            return len(self._text(args[0]))
        if f == "overlap":
            return len(low(args[0]) & low(args[1]))
        if f == "share":
            a = low(args[0])
            return len(a & low(args[1])) / len(a) if a else 0.0
        if f == "jaccard":
            a, b = low(args[0]), low(args[1])
            return len(a & b) / len(a | b) if a | b else 0.0
        if f == "only_in":
            return len(low(args[0]) - low(args[1]))
        if f in ("count", "has"):
            if len(args) < 2 or not all(isinstance(w, str) for w in args[1:]):
                raise FormulaError(f"{f}(x, 'word', ...) needs a text and words in quotes")
            ws = {w.lower() for w in args[1:]}
            c = sum(1 for t in self._text(args[0]) if t.lower() in ws)
            return c if f == "count" else float(c > 0)
        if f == "neg":
            return sum(1 for t in self._text(args[0]) if t.lower() in NEGATIONS)
        if f == "unk":
            if g is None:
                return 0
            return sum(1 for t in self._text(args[0]) if g.row(t, ecfg["fallback_lower"]) is None)
        if f == "cos":
            if g is None:
                return 0.0
            va, vb = mean_vec(self._text(args[0]), g, ecfg), mean_vec(self._text(args[1]), g, ecfg)
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            return float(va @ vb / (na * nb)) if na and nb else 0.0
        if f == "min":
            return min(args)
        if f == "max":
            return max(args)
        if f == "abs":
            return abs(args[0])
        if f == "log":
            return math.log1p(max(0.0, args[0]))
        raise FormulaError(f)


def mean_vec(toks, g, ecfg):
    vs = [v for v in (g.vec(t, ecfg["fallback_lower"]) for t in toks) if v is not None]
    return np.mean(vs, 0) if vs else np.zeros(g.dim, dtype=np.float32)


def feature_list(inputs):
    return [f for f in (inputs.get("text_features") or []) if not f.get("off")]


def check_features(inputs):
    errs, seen = [], set()
    for i, f in enumerate(inputs.get("text_features") or []):
        if f.get("off"):
            continue
        try:
            nm = f.get("name") or ""
            if not nm.isidentifier() or nm in seen:
                raise FormulaError(f"the name '{nm}' is empty, not a plain name, or used twice")
            seen.add(nm)
            TextFormula(f.get("formula", ""))
        except FormulaError as e:
            errs.append({"index": i, "name": f.get("name"), "msg": str(e)})
    return errs


def feature_values(rows_tok, feats, g, ecfg):
    """[n_rows, 2 candidates, n_features]: every feature for h = h1 and h = h2."""
    fs = [TextFormula(f["formula"]) for f in feats]
    out = np.zeros((len(rows_tok), 2, len(fs)), dtype=np.float32)
    for i, r in enumerate(rows_tok):
        for c, (h, other) in enumerate(((r["h1"], r["h2"]), (r["h2"], r["h1"]))):
            env = {"o1": r["o1"], "o2": r["o2"], "h": h, "other": other}
            for j, f in enumerate(fs):
                out[i, c, j] = f(env, g, ecfg)
    return out
