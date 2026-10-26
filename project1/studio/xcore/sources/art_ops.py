"""Blocks for text (ART): recurrent layers, joining sentences into one sequence, comparison features and the output contracts.

Words arrive as sets ([B, T, D] + padding mask) in sentence order, padding at the end. A set keeps its order, so the
set blocks (dense layer, summarise, transformer) work on words too; the RNN reads them one after the other.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from xcore import graph as G
from xcore.graph import GraphError, Meta, Op, Param, Port, SetVal, need, register


def compact(x, mask):
    """Move the real items of every row to the front (stable), padding to the back."""
    order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices
    x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return x, torch.gather(mask, 1, order)


@register
class RNN(Op):
    key, label, category, source = "rnn", "Recurrent layer (LSTM / GRU)", "mix", "art"
    inputs = (Port("in", ("set",), help="words in sentence order"),)
    outputs = (Port("seq", ("set",), help="one output per word (what the RNN 'thinks' after reading up to that word)"),
               Port("last", ("vec",), help="the state after the last real word (both directions joined if bidirectional)"))
    params = (Param("cell", "choice", "gru", "GRU: 2 gates, fewer weights, trains faster. LSTM: 3 gates + a separate memory cell. "
                                             "rnn: the plain Elman RNN (forgets quickly; for comparison).", choices=("gru", "lstm", "rnn")),
              Param("hidden", "int", 128, "Size of the memory (hidden state) per direction.", 4, 1024),
              Param("layers", "int", 2, "Stacked RNNs: the second one reads the outputs of the first.", 1, 4),
              Param("bidirectional", "bool", False, "Also read the sequence backwards (a second RNN); outputs of both directions are joined."),
              Param("dropout", "float", 0.0, "Dropout between the stacked layers (training only; needs 2+ layers).", 0, 0.8, step=0.05))
    doc = {"what": "Reads the words one at a time and updates a memory (hidden state) after each word. The state after the last word "
                   "summarises the whole sequence; the per-word outputs say what it knew at every point.",
           "why": "Unlike a mean of word vectors, it sees word ORDER: 'he left before she came' differs from 'she came before he left'.",
           "when": "Architecture 2: words of O1 -> h -> O2 joined into one sequence, then the last state goes to the classifier.",
           "watch": "Padding is skipped (packed sequences), so the last state is the last REAL word. Long sequences make plain RNNs forget the "
                    "beginning; GRU / LSTM gates help. Training is slower than a mean (one step per word)."}

    def infer(self, p, ins):
        m = need(ins, "in", ("set",))
        h = p["hidden"] * (2 if p["bidirectional"] else 1)
        return {"seq": Meta("set", h, name=m.name), "last": Meta("vec", h)}

    def build(self, p, ins):
        cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[p["cell"]]
        return cls(ins["in"].dim, p["hidden"], num_layers=p["layers"], batch_first=True, bidirectional=p["bidirectional"],
                   dropout=p["dropout"] if p["layers"] > 1 else 0.0)

    def forward(self, mod, p, ins, ctx):
        s = ins["in"]
        x, mask = s.x, s.mask if s.mask is not None else torch.ones(s.x.shape[:2], dtype=torch.bool, device=s.x.device)
        lengths = mask.sum(1).clamp(min=1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = mod(packed)
        seq, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        h = h[0] if isinstance(h, tuple) else h                     # LSTM: (h, c)
        dirs = 2 if p["bidirectional"] else 1
        last = torch.cat([h[-dirs + k] for k in range(dirs)], -1)    # top layer, forward (+ backward)
        return {"seq": SetVal(seq, mask, None), "last": last}

    def flops(self, p, ins, outs, cs):
        gates = {"gru": 3, "lstm": 4, "rnn": 1}[p["cell"]]
        n = cs.get(ins["in"].name, 10)
        d, h, t = ins["in"].dim, p["hidden"], 0
        for layer in range(p["layers"]):
            t += gates * (d * h + h * h) * n
            d = h * (2 if p["bidirectional"] else 1)
        return t * (2 if p["bidirectional"] else 1)


@register
class SeqJoin(Op):
    key, label, category, source = "seq_join", "Join sentences (one sequence)", "mix", "art"
    inputs = (Port("in", ("set",), multi=True, help="sentences in the order they should be read (the editor shows the order)"),)
    outputs = (Port("out", ("set",)),)
    params = (Param("marker", "bool", True, "Put one all-zero 'word' between the sentences, so a reader can tell where one ends."),
              Param("part_ids", "bool", False, "Add one number per sentence to every word (1 if the word comes from that sentence): "
                                                "the model knows which sentence each word belongs to."))
    doc = {"what": "Glues several word sequences into one, in the order of the edges (e.g. O1, h, O2: the story in its natural order). "
                   "Padding is moved to the end, so the words follow each other directly.",
           "why": "An RNN reads one sequence. Reading O1 -> h -> O2 lets it check whether the hypothesis leads from the beginning to the end.",
           "when": "Before a recurrent layer or a transformer over all words.",
           "watch": "All parts need the same width (the same word vectors). The order of the edges is the reading order."}

    def infer(self, p, ins):
        ms = need(ins, "in", ("set",))
        ms = ms if isinstance(ms, list) else [ms]
        dims = {m.dim for m in ms}
        if len(dims) > 1:
            raise GraphError(f"all sentences need the same width, got {sorted(dims)}")
        return {"out": Meta("set", dims.pop() + (len(ms) if p["part_ids"] else 0), name="+".join(m.name or "?" for m in ms))}

    def forward(self, mod, p, ins, ctx):
        xs = ins["in"] if isinstance(ins["in"], list) else [ins["in"]]
        B, D = xs[0].x.shape[0], xs[0].x.shape[-1]
        dev = xs[0].x.device
        parts, masks = [], []
        for k, s in enumerate(xs):
            x = s.x
            m = s.mask if s.mask is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
            if p["part_ids"]:
                one = torch.zeros(B, x.shape[1], len(xs), device=dev, dtype=x.dtype)
                one[..., k] = 1.0
                x = torch.cat([x, one], -1)
            parts.append(x)
            masks.append(m)
            if p["marker"] and k < len(xs) - 1:
                parts.append(torch.zeros(B, 1, x.shape[-1], device=dev, dtype=x.dtype))
                masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        x, mask = compact(torch.cat(parts, 1), torch.cat(masks, 1))
        n = int(mask.sum(1).max()) if mask.numel() else 0
        return {"out": SetVal(x[:, :max(1, n)], mask[:, :max(1, n)], None)}


@register
class PairFeatures(Op):
    key, label, category, source = "pair_features", "Compare two vectors", "set", "art"
    inputs = (Port("a", ("vec",), help="e.g. the mean vector of the observations"), Port("b", ("vec",), help="e.g. the mean vector of the hypothesis"))
    outputs = (Port("out", ("vec",)),)
    params = (Param("a", "bool", True, "include a itself"), Param("b", "bool", True, "include b itself"),
              Param("product", "bool", True, "a * b, number by number (large where both agree)"),
              Param("difference", "bool", True, "|a - b|, number by number (large where they differ)"),
              Param("cosine", "bool", False, "one number: the cosine similarity of a and b"))
    doc = {"what": "Builds comparison features of two vectors of the same width: [a, b, a*b, |a-b|] (+ cosine).",
           "why": "A small classifier struggles to compute 'how similar are a and b' from [a, b] alone; handing it the product and the "
                  "difference makes that easy. Known trick from InferSent (Conneau et al. 2017, cite it if you use it).",
           "when": "Arch 1: compare the hypothesis with each observation before the MLP.",
           "watch": "Nothing here is learned, so it is allowed in 'train only the classifier'. It multiplies the input width (4x)."}

    def infer(self, p, ins):
        a, b = need(ins, "a", ("vec",)), need(ins, "b", ("vec",))
        if a.dim != b.dim:
            raise GraphError(f"a and b need the same width ({a.dim} vs {b.dim})")
        d = a.dim * (p["a"] + p["b"] + p["product"] + p["difference"]) + (1 if p["cosine"] else 0)
        if d == 0:
            raise GraphError("switch on at least one part")
        return {"out": Meta("vec", d)}

    def forward(self, mod, p, ins, ctx):
        a, b = ins["a"], ins["b"]
        parts = ([a] if p["a"] else []) + ([b] if p["b"] else []) + ([a * b] if p["product"] else []) + ([(a - b).abs()] if p["difference"] else [])
        if p["cosine"]:
            parts.append(torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-6).unsqueeze(-1))
        return {"out": torch.cat(parts, -1)}


@register
class OutScore(Op):
    key, label, category, source = "out_score", "Output: score of this hypothesis", "output", "art"
    inputs = (Port("score", ("vec",), help="one number"),)
    outputs = ()
    params = ()
    doc = {"what": "One number: how plausible the hypothesis is that the model is looking at. The model runs once for h1 and once for h2 "
                   "(same weights); the two scores go through a softmax and the higher one is the answer.",
           "why": "Treats both hypotheses exactly alike by design (swapping them swaps the scores), so the model cannot learn a 'h1 is usually right' bias.",
           "when": "I/O format 'score each hypothesis'.", "watch": "Needs exactly one number."}

    def infer(self, p, ins):
        m = need(ins, "score", ("vec",))
        if m.dim != 1:
            raise GraphError(f"needs exactly 1 number, gets {m.dim}")
        return {}

    def forward(self, mod, p, ins, ctx):
        ctx["out"]["score"] = ins["score"][:, 0]
        return {}


@register
class OutClass(Op):
    key, label, category, source = "out_class", "Output: which hypothesis (classes)", "output", "art"
    inputs = (Port("logits", ("vec",), help="one score per class: [h1, h2]"),)
    outputs = ()
    params = ()
    doc = {"what": "Two numbers in one pass: the score for 'h1 is the plausible one' and for 'h2 is'. Softmax + cross-entropy, the higher wins.",
           "why": "The classic classification head: the model sees everything at once and can compare h1 and h2 directly.",
           "when": "I/O format 'all in one'.",
           "watch": "The model has to learn that swapping h1 and h2 swaps the answer; 'swap augmentation' in Training helps."}

    def infer(self, p, ins):
        m = need(ins, "logits", ("vec",))
        if m.dim != 2:
            raise GraphError(f"needs 2 numbers (h1, h2), gets {m.dim}")
        return {}

    def forward(self, mod, p, ins, ctx):
        ctx["out"]["logits"] = ins["logits"]
        return {}


OPS = {k: G.OPS[k] for k in ("rnn", "seq_join", "pair_features", "out_score", "out_class")}
