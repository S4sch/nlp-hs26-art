// Flow (step by step) views for text: words -> vectors -> pooled sentence vectors / RNN states -> scores, on a real example.
import { esc, fmt } from "./util.js";
import { decisionHtml, storyHtml, vstrip } from "./textui.js";

const rowsHtml = (rows, labels, k = 64) => `<div class="tx-vecs">${rows.map((r, i) => `<div class="tx-vrow"><span class="mono tiny">${esc(labels?.[i] ?? "#" + i)}</span>${vstrip(r.slice(0, k))}</div>`).join("")}</div>`;

function wordsFor(d, port, n) {
  const w = d.base?.words || {};
  if (w[port] && w[port].length === n) return w[port];
  for (const s of [["o1", "h", "o2"], ["o1", "h1", "h2", "o2"]]) {
    if (!s.every((f) => w[f])) continue;
    const plain = s.flatMap((f) => w[f]);
    if (plain.length === n) return plain;
    const marked = s.flatMap((f, i) => (i < s.length - 1 ? [...w[f], "·"] : w[f]));
    if (marked.length === n) return marked;
  }
  return null;
}

function inputView(ED, n, d, el) {
  const v = d.outs.out, port = n.params.port;
  if (v.kind === "vec") {
    el.innerHTML = `${storyHtml(d.base)}<p class="small ink2">Your own text features for this hypothesis (standardised: 0 = average on train, ±1 = one spread away):</p>${vstrip(v.values)}
      <div class="small">${(ED.d.inputs.text_features || []).filter((f) => !f.off).map((f, i) => `<code>${esc(f.name)}</code> = ${fmt(v.values[i], 2)}`).join(" · ")}</div>`;
    return;
  }
  const words = wordsFor(d, port, v.n);
  el.innerHTML = `${storyHtml(d.base)}
    <p class="small ink2">The sentence <b>${esc(port)}</b> after the pipeline: ${v.n} words, each replaced by its frozen GloVe vector of ${v.dim} numbers (one row per word, first 64 shown;
      red = positive, blue = negative). This table is all the model ever sees of the sentence — the letters are gone.</p>
    ${rowsHtml(v.rows, words)}
    <div class="learn" style="margin-top:10px"><b>Why vectors:</b> words with similar uses get similar rows ("ran" ≈ "walked"), so what the model learns for one word carries over to its neighbours.
      The price: one fixed vector per word, whatever the context ("bank" of a river = "bank" with money).</div>`;
}

function poolView(ED, n, d, el) {
  const inp = d.ins[0].value, out = d.outs.out, mode = d.params.mode;
  const words = wordsFor(d, d.ins[0].from.replace(/^in_/, ""), inp.n) || wordsFor(d, "", inp.n);
  const k = 64;
  el.innerHTML = `<p class="small ink2">${inp.n} word vectors → <b>one</b> vector: ${mode === "mean" ? "for each of the " + inp.dim + " positions, the average over the words" : mode === "max" ? "for each position, the largest value of any word" : mode === "mean+max" ? "the average and the largest value side by side (" + out.dim + " numbers)" : "the sum over the words"}.
    Padding is ignored.</p>${rowsHtml(inp.rows, words, k)}
    <div class="tx-vrow mean" style="margin-top:4px"><span class="mono tiny"><b>= ${esc(mode)}</b></span>${vstrip(out.values.slice(0, k))}</div>
    ${mode === "mean+max" ? `<div class="tx-vrow mean"><span class="mono tiny"><b>max part</b></span>${vstrip(out.values.slice(inp.dim, inp.dim + k))}</div>` : ""}
    <div class="learn" style="margin-top:10px"><b>What gets lost:</b> the order of the words ("he left before she came" = "she came before he left") and, in a mean, single words in long sentences.
      Frequent small words (the, a, was) pull every mean towards the same place — compare h1 and h2 in the Inputs tab.</div>`;
}

function joinView(ED, n, d, el) {
  const out = d.outs.out;
  const words = wordsFor(d, "", out.n);
  el.innerHTML = `<p class="small ink2">${d.ins.length} sentences glued into one sequence of ${out.n} items, in this order: ${d.ins.map((x) => `<code>${esc(x.from)}</code>`).join(" → ")}
    ${d.params.marker ? " (· = an all-zero marker between sentences)" : ""}${d.params.part_ids ? `; the last ${d.ins.length} numbers of every row say which sentence it came from` : ""}.</p>
    ${rowsHtml(out.rows, words)}
    <div class="learn" style="margin-top:10px"><b>Why this order:</b> O1 → h → O2 is the story as it happened. A reader that goes left to right can check whether the hypothesis
      leads from the beginning to the end.</div>`;
}

function rnnView(ED, n, d, el) {
  const inp = d.ins[0].value, seq = d.outs.seq, last = d.outs.last, p = d.params;
  const words = wordsFor(d, "", seq.n);
  const dirs = p.bidirectional ? 2 : 1;
  el.innerHTML = `<p class="small ink2">A ${p.layers}-layer ${p.cell.toUpperCase()}${p.bidirectional ? ", bidirectional," : ""} reads the ${inp.n} items one after another. After each word it updates its memory
    (${p.hidden} numbers${dirs > 1 ? " per direction" : ""}). Each row below is the memory <b>after</b> that word (top layer${dirs > 1 ? ", forward | backward" : ""}).</p>
    ${rowsHtml(seq.rows, words, 64)}
    <div class="tx-vrow mean" style="margin-top:4px"><span class="mono tiny"><b>last</b></span>${vstrip(last.values.slice(0, 64))}</div>
    <div class="tiny muted">"last" = the state after the last real word${dirs > 1 ? " (forward) joined with the backward reader's state after the first word" : ""}: ${last.dim} numbers, the summary that goes on.</div>
    <div class="learn" style="margin-top:10px"><b>Watch the rows change:</b> words that matter for the story move the memory a lot; filler words barely change it (with trained weights).
      ${p.cell === "gru" ? "A GRU decides per number how much to keep (update gate) and how much of the old memory to use for the new candidate (reset gate)." : p.cell === "lstm" ? "An LSTM has a separate cell memory with forget / input / output gates: it can carry a fact over many words." : "A plain RNN overwrites its memory a little at every word: early words fade quickly."}</div>`;
}

function pairView(ED, n, d, el) {
  const a = d.ins.find((x) => x.port === "a")?.value, b = d.ins.find((x) => x.port === "b")?.value, out = d.outs.out, p = d.params;
  const D = a.dim, k = 64;
  const parts = [];
  if (p.a) parts.push(["a", a.values]);
  if (p.b) parts.push(["b", b.values]);
  if (p.product) parts.push(["a × b", a.values.map((x, i) => x * b.values[i])]);
  if (p.difference) parts.push(["|a − b|", a.values.map((x, i) => Math.abs(x - b.values[i]))]);
  const dot = a.values.reduce((s, x, i) => s + x * b.values[i], 0), na = Math.hypot(...a.values), nb = Math.hypot(...b.values);
  el.innerHTML = `<p class="small ink2">a from <code>${esc(d.ins.find((x) => x.port === "a")?.from)}</code>, b from <code>${esc(d.ins.find((x) => x.port === "b")?.from)}</code> (${D} numbers each; first ${k} shown):</p>
    ${parts.map(([l, v]) => `<div class="tx-vrow"><span class="mono tiny">${esc(l)}</span>${vstrip(v.slice(0, k))}</div>`).join("")}
    <div class="small" style="margin-top:6px">cosine(a, b) = <b>${fmt(na && nb ? dot / (na * nb) : 0, 3)}</b>${p.cosine ? " (added as the last number)" : ""} → ${out.dim} numbers go on.</div>
    <div class="learn" style="margin-top:10px"><b>Why:</b> a × b is large where both vectors agree, |a − b| where they differ. A 2-layer classifier would need many weights to compute that
      from [a, b] itself; handed over, it can use "how similar is h to the ending" directly.</div>`;
}

function scoreView(ED, n, d, el) {
  const dec = d.decision;
  el.innerHTML = `${storyHtml(d.base)}${dec ? decisionHtml(dec, d.base) : ""}
    <p class="small ink2">${n.op === "out_score" ? "This pass produced one number for the hypothesis marked 'scored now'. The partner pass (same weights) scored the other one; softmax over the two gives the probabilities. Training pushes the plausible one's score up and the other's down (cross-entropy)."
      : "Two numbers in one pass: h1 right / h2 right. Softmax turns them into probabilities; training raises the right one (cross-entropy)."}</p>
    ${n.op === "out_score" ? `<div class="tiny muted">Position buttons: 1/2 are h1/h2 of the same example, 3/4 the next example, …</div>` : ""}`;
}

export const TEXT_VIEWS = { input: inputView, set_pool: poolView, seq_join: joinView, rnn: rnnView, pair_features: pairView, out_score: scoreView, out_class: scoreView };
