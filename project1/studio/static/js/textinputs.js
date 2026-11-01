// Inputs tab for text sources (ART): the preprocessing pipeline, the frozen word vectors and own text features, each change
// shown live on real examples and on the whole training split. Nothing here is learned: it decides which numbers the model gets.
import { $, $$, api, debounce, esc, fmt, fmtN, icon, toast } from "./util.js";

let X = null;  // { ED, markDirty, analyseSoon, split, index, field, word, feat, stats, prev }

const FIELD_NAMES = { o1: "O1 · beginning", h1: "h1", h2: "h2", o2: "O2 · end" };

function cfg() {
  const inp = X.ED.d.inputs;
  const src = X.ED.src;
  inp.text ||= {};
  inp.text.steps ||= JSON.parse(JSON.stringify(src.default_steps));
  inp.text.embedding = { name: "glove", dim: 300, unknown: "zero", fallback_lower: true, ...(inp.text.embedding || {}) };
  inp.text.format ||= "score_each";
  inp.text_features ||= [];
  return inp;
}

export function tabTextInputs(body, ED, markDirty, analyseSoon) {
  X = { ED, markDirty, analyseSoon, split: X?.split || "validation", index: X?.index ?? 0, field: X?.field || "h1", word: null, feat: X?.feat ?? null,
        stats: null, prev: null, statsSplit: X?.statsSplit || "train", featSplit: X?.featSplit || "train", stepSel: X?.stepSel ?? null, req: 0 };
  cfg();
  body.style.overflow = "hidden";
  body.innerHTML = `<div class="editor wide-right tx" style="height:100%"><div class="ed-left" id="txleft" style="overflow:auto"></div>
    <div class="ed-mid" style="overflow:auto"><div style="padding:14px 18px" id="txmid"></div></div><div class="ed-right" id="txright" style="overflow:auto"></div></div>`;
  renderLeft();
  renderMid();
  renderRight();
  refreshSoon(true);
}

function changed(rebuildLeft = true) {
  X.markDirty();
  X.analyseSoon();
  if (rebuildLeft) renderLeft();
  refreshSoon(true);
}

// ---------------------------------------------------------------------------------------------- left: the recipe
function renderLeft() {
  const inp = cfg(), t = inp.text, src = X.ED.src, S = src.steps;
  const tokAt = t.steps.findIndex((s) => s.op === "tokenize" && !s.off);
  const stepHtml = (s, i) => {
    const d = S[s.op] || { label: s.op, params: [], help: "" };
    const where = d.where === "split" ? "split" : i < tokAt || tokAt < 0 ? "sentence" : "words";
    const params = (d.params || []).map((p) => {
      const v = s[p.key] ?? p.default;
      if (p.type === "choice") return `<label class="f tiny">${esc(p.key)}<select data-sp="${i}:${p.key}">${p.choices.map((c) => `<option ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>`;
      if (p.type === "int") return `<label class="f tiny">${esc(p.key)}<input type="number" data-sp="${i}:${p.key}" value="${esc(v)}" min="${p.min ?? ""}" max="${p.max ?? ""}" style="width:80px"></label>`;
      return `<label class="f tiny">${esc(p.key)}<textarea rows="2" data-sp="${i}:${p.key}">${esc(v)}</textarea></label>`;
    }).join("");
    return `<div class="tx-step ${s.off ? "off" : ""} ${X.stepSel === i ? "on" : ""}" data-si="${i}">
      <div class="row" style="gap:6px"><input type="checkbox" data-son="${i}" ${s.off ? "" : "checked"} ${s.op === "tokenize" ? "disabled" : ""} title="switch on / off">
        <b class="small" style="flex:1">${esc(d.label)}</b><span class="tx-where ${where}">${where}</span>
        <button class="icb sm" data-sup="${i}" title="move up">▲</button><button class="icb sm" data-sdn="${i}" title="move down">▼</button>
        ${s.op === "tokenize" ? "" : `<button class="icb sm" data-sdel="${i}" title="remove">✕</button>`}</div>
      <div class="tiny muted" style="margin:3px 0 0 22px">${esc(d.help)}</div>
      ${params ? `<div style="margin:4px 0 0 22px" class="col">${params}</div>` : ""}
      <div class="tiny" style="margin:3px 0 0 22px" data-simp="${i}"></div></div>`;
  };
  const fmtOpt = Object.entries(src.formats).map(([k, f]) => `<label class="tx-radio ${t.format === k ? "on" : ""}"><input type="radio" name="txfmt" value="${k}" ${t.format === k ? "checked" : ""}>
    <div><b class="small">${esc(f.label)}</b><div class="tiny muted">${esc(f.help)}</div><div class="tiny">inputs: ${f.ports.map((p) => `<code>${p}</code>`).join(" ")} · output block: <code>${esc(f.output)}</code></div></div></label>`).join("");
  const feats = inp.text_features.map((f, i) => `<div class="tx-feat ${X.feat === i ? "on" : ""} ${f.off ? "off" : ""}" data-fi="${i}">
      <div class="row" style="gap:6px"><input type="checkbox" data-fon="${i}" ${f.off ? "" : "checked"}><input type="text" class="mono" data-fname="${i}" value="${esc(f.name)}" style="flex:1;min-width:0">
        <button class="icb sm" data-fdel="${i}" title="remove">✕</button></div>
      <input type="text" class="mono" data-fform="${i}" value="${esc(f.formula)}" style="width:100%;margin-top:4px" spellcheck="false">
      <div class="tiny" data-ferr="${i}"></div></div>`).join("");
  $("#txleft").innerHTML = `<div class="pal">
    <div class="sect-h" style="margin-top:2px">1 · I/O format</div><div class="col" style="gap:6px">${fmtOpt}</div>
    <div class="sect-h">2 · Preprocessing (top to bottom)</div>
    <div class="tiny muted" style="margin-bottom:6px">Click a step to see what it does to the example. Untick it to see the difference. Steps above <b>Tokenize</b> work on the sentence, below it on the word list.</div>
    <div class="col" style="gap:6px">${t.steps.map(stepHtml).join("")}</div>
    <div class="row" style="margin-top:6px"><select id="txadd"><option value="">+ add a step…</option>${Object.entries(S).filter(([k]) => k !== "tokenize").map(([k, d]) => `<option value="${k}">${esc(d.label)}</option>`).join("")}</select>
      <button class="btn ghost sm" id="txreset" title="back to the default pipeline">reset</button></div>
    <div class="sect-h">3 · Word vectors (frozen)</div>
    <label class="f small">GloVe 6B size<select id="txdim">${[50, 100, 200, 300].map((d) => `<option ${+t.embedding.dim === d ? "selected" : ""}>${d}</option>`).join("")}</select>
      <span class="tiny muted" style="font-weight:400">Numbers per word. More = more detail, and more weights in the first layer that reads them.</span></label>
    <label class="f small">Words without a vector become<select id="txunk">${Object.entries(src.unknown).map(([k, h]) => `<option value="${k}" ${t.embedding.unknown === k ? "selected" : ""}>${k}: ${esc(h)}</option>`).join("")}</select></label>
    <label class="row small"><input type="checkbox" id="txlow" ${t.embedding.fallback_lower ? "checked" : ""}> if a word is missing, try it lowercased</label>
    <div class="sect-h">4 · Own text features <span class="tiny muted" style="text-transform:none;letter-spacing:0">(optional)</span></div>
    <div class="tiny muted" style="margin-bottom:6px">One number per formula and hypothesis, standardised on train, offered as the <code>features</code> input (connect it in the Model tab).
      They go beyond "word vectors + classifier": keep a run without them as your official Arch 1.</div>
    <div class="col" style="gap:6px">${feats || `<div class="tiny muted">none yet</div>`}</div>
    <div class="row" style="margin-top:6px"><button class="btn sm" id="txfadd">${icon("plus")} formula</button>
      <select id="txidea"><option value="">ideas…</option>${src.feature_ideas.map((f, i) => `<option value="${i}">${esc(f.name)}: ${esc(f.formula)}</option>`).join("")}</select></div>
    <details class="box" style="margin-top:8px"><summary class="small">Formula reference</summary>
      <div class="tiny" style="margin-top:4px">${Object.entries(src.text_vars).map(([k, v]) => `<div><code>${k}</code> ${esc(v)}</div>`).join("")}</div>
      <div class="tiny" style="margin-top:6px">${src.functions.map((f) => `<div><code>${esc(f.signature)}</code> ${esc(f.help)}</div>`).join("")}</div>
      <div class="tiny muted" style="margin-top:4px">+ − × ÷, comparisons (&gt; &lt; ==) give 1 / 0. Words in quotes: <code>count(h, 'not', 'never')</code>.</div></details>
  </div>`;
  bindLeft();
  showImpact();
}

function bindLeft() {
  const inp = cfg(), t = inp.text;
  $$("[name=txfmt]").forEach((r) => r.onchange = () => {
    t.format = r.value;
    toast(`format: ${X.ED.src.formats[r.value].label}. The graph needs the inputs ${X.ED.src.formats[r.value].ports.join(", ")} and ${X.ED.src.formats[r.value].output}: easiest with the matching template.`);
    changed();
  });
  $$("[data-son]").forEach((c) => c.onchange = () => { t.steps[+c.dataset.son].off = !c.checked; changed(); });
  $$("[data-sup]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); move(+b.dataset.sup, -1); });
  $$("[data-sdn]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); move(+b.dataset.sdn, 1); });
  $$("[data-sdel]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); t.steps.splice(+b.dataset.sdel, 1); X.stepSel = null; changed(); });
  $$("[data-sp]").forEach((el) => el.onchange = () => {
    const [i, k] = el.dataset.sp.split(":");
    t.steps[+i][k] = el.type === "number" ? +el.value : el.value;
    changed(false);
  });
  $$(".tx-step").forEach((el) => el.onclick = (e) => {
    if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName)) return;
    X.stepSel = X.stepSel === +el.dataset.si ? null : +el.dataset.si;
    $$(".tx-step").forEach((x) => x.classList.toggle("on", +x.dataset.si === X.stepSel));
    renderMid();
  });
  $("#txadd").onchange = (e) => {
    const op = e.target.value; if (!op) return;
    const d = X.ED.src.steps[op], st = { op };
    for (const p of d.params || []) st[p.key] = p.default;
    const tok = t.steps.findIndex((s) => s.op === "tokenize");
    if (d.where === "text") t.steps.splice(Math.max(0, tok), 0, st); else t.steps.push(st);
    X.stepSel = t.steps.indexOf(st);
    changed();
  };
  $("#txreset").onclick = () => { if (!confirm("Back to the default pipeline?")) return; t.steps = JSON.parse(JSON.stringify(X.ED.src.default_steps)); changed(); };
  $("#txdim").onchange = (e) => { t.embedding.dim = +e.target.value; changed(false); };
  $("#txunk").onchange = (e) => { t.embedding.unknown = e.target.value; changed(false); };
  $("#txlow").onchange = (e) => { t.embedding.fallback_lower = e.target.checked; changed(false); };
  const F = inp.text_features;
  $("#txfadd").onclick = () => { F.push({ name: `feature_${F.length + 1}`, formula: "share(h, o2)" }); X.feat = F.length - 1; changed(); renderRight(); };
  $("#txidea").onchange = (e) => {
    const k = e.target.value; if (k === "") return;
    const f = X.ED.src.feature_ideas[+k];
    let nm = f.name; for (let j = 2; F.some((x) => x.name === nm); j++) nm = `${f.name}_${j}`;
    F.push({ name: nm, formula: f.formula, description: f.why });
    X.feat = F.length - 1; toast(f.why); changed(); renderRight();
  };
  $$("[data-fon]").forEach((c) => c.onchange = () => { F[+c.dataset.fon].off = !c.checked; changed(); });
  $$("[data-fdel]").forEach((b) => b.onclick = () => { F.splice(+b.dataset.fdel, 1); X.feat = null; changed(); renderRight(); });
  $$("[data-fname]").forEach((el) => el.oninput = () => { F[+el.dataset.fname].name = el.value; X.markDirty(); X.analyseSoon(); });
  const featSoon = debounce(() => { refreshSoon(false); renderRight(); }, 500);
  $$("[data-fform]").forEach((el) => el.oninput = () => { F[+el.dataset.fform].formula = el.value; X.feat = +el.dataset.fform; X.markDirty(); X.analyseSoon(); featSoon(); });
  $$(".tx-feat").forEach((el) => el.onclick = (e) => {
    if (e.target.tagName === "BUTTON") return;
    if (X.feat !== +el.dataset.fi) { X.feat = +el.dataset.fi; $$(".tx-feat").forEach((x) => x.classList.toggle("on", x === el)); renderRight(); }
  });
}

function move(i, dir) {
  const s = cfg().text.steps, j = i + dir;
  if (j < 0 || j >= s.length) return;
  [s[i], s[j]] = [s[j], s[i]];
  X.stepSel = j;
  changed();
}

export function updateTextErrors(A) {
  if (!X || !$("#txleft")) return;
  const errs = A?.feature_errors || [];
  $$("[data-ferr]").forEach((el) => {
    const e = errs.find((x) => x.index === +el.dataset.ferr);
    el.innerHTML = e ? `<span class="l">${esc(e.msg)}</span>` : "";
  });
  const pe = (A?.errors || []).filter((e) => !e.node && e.msg.startsWith("Inputs:"));
  const box = $("#txperr");
  if (box) box.innerHTML = pe.map((e) => `<div class="errbox">${esc(e.msg)}</div>`).join("");
}

function showImpact() {
  const st = X.stats;
  if (!st?.steps) return;
  const on = cfg().text.steps.map((s, i) => [s, i]).filter(([s]) => !s.off);
  on.forEach(([s, i], k) => {
    const c = st.steps[k], el = $(`[data-simp="${i}"]`);
    if (!c || !el || c.op !== s.op) return;
    el.innerHTML = `<span class="muted">changes</span> <b>${Math.round(c.changed_share * 1000) / 10} %</b> <span class="muted">of the sentences${c.tokens_per_sentence ? ` · then ${c.tokens_per_sentence} words per sentence` : ""}</span>`;
  });
}

// ---------------------------------------------------------------------------------------------- middle: one real example
const refreshSoon = debounce((withStats) => refresh(withStats), 350);

async function refresh(withStats) {
  const inp = cfg();
  const req = ++X.req;
  try {
    const p = await api("/api/text/preview", { inputs: inp, split: X.split, index: X.index, vectors: true, max_dims: 64 });
    if (req !== X.req) return;
    X.prev = p;
    renderMid();
  } catch (e) { $("#txmid").innerHTML = `<div class="errbox">${esc(e.message)}</div>`; }
  if (withStats) {
    try {
      const s = await api("/api/text/stats", { inputs: inp, split: X.statsSplit });
      if (s.error) throw new Error(s.error);
      X.stats = s; showImpact();
    } catch (e) { X.stats = { error: e.message }; }
    renderRight();
  }
}
function chips(tokens, { removed = null, clickable = true } = {}) {
  return tokens.map((t) => {
    const w = typeof t === "string" ? { w: t, known: true } : t;
    const cls = removed?.has(w.w) ? "gone" : !w.known ? "unk" : w.via_lower ? "low" : "";
    return `<span class="tx-tok ${cls}" ${clickable ? `data-word="${esc(w.w)}"` : ""} title="${!w.known ? "no word vector" : w.via_lower ? "found lowercased" : w.rank != null ? `GloVe row ${w.rank} (lower = more frequent)` : ""}">${esc(w.w)}</span>`;
  }).join("");
}

function renderMid() {
  const p = X.prev, el = $("#txmid");
  if (!el) return;
  if (!p) { el.innerHTML = `<div class="muted small">loading the example…</div>`; return; }
  const fmtK = cfg().text.format;
  const lab = p.label;  // 0 = h1 plausible
  const sent = (f) => {
    const good = (f === "h1" && lab === 0) || (f === "h2" && lab === 1);
    const bad = (f === "h1" && lab === 1) || (f === "h2" && lab === 0);
    return `<div class="tx-sent ${X.field === f ? "on" : ""}" data-field="${f}"><div class="tx-lab">${esc(FIELD_NAMES[f])}${good ? ` <span class="pill ok">plausible</span>` : bad ? ` <span class="pill bad">not plausible</span>` : ""}</div>
      <div class="tx-raw">${esc(p.fields[f].raw)}</div><div class="tx-toks">${chips(p.fields[f].tokens)}</div></div>`;
  };
  const F = p.fields[X.field];
  // step by step for the selected sentence
  let prevToks = null;
  const stages = F.stages.map((s, k) => {
    if (s.off) return `<div class="tx-stage off"><span class="num">${k + 1}</span><span class="small muted">${esc(s.label || s.op)} (off)</span></div>`;
    const sel = X.stepSel != null && cfg().text.steps[X.stepSel] && cfg().text.steps[X.stepSel].op === s.op && k === X.stepSel;
    let body;
    if (typeof s.value === "string") body = `<span class="mono small">${esc(s.value)}</span>`;
    else {
      const now = new Set(s.value);
      const gone = prevToks ? prevToks.filter((w) => !now.has(w)) : [];
      body = chips(s.value, { clickable: true }) + (gone.length ? ` <span class="tiny muted">removed:</span> ${chips(gone, { removed: new Set(gone), clickable: false })}` : "");
      prevToks = s.value;
    }
    return `<div class="tx-stage ${s.changed ? "chg" : ""} ${sel ? "on" : ""}"><span class="num">${k + 1}</span><div><div class="tiny"><b>${esc(s.label || s.op)}</b>${s.changed ? "" : ` <span class="muted">— no change here</span>`}</div>${body}</div></div>`;
  }).join("");
  const sims = p.sims;
  const sv = (a, b) => sims[a < b ? `${a}-${b}` : `${b}-${a}`];
  const order = ["o1", "h1", "h2", "o2"];
  const simTable = `<table class="tx-sim"><tr><th></th>${order.map((f) => `<th>${f}</th>`).join("")}</tr>${order.map((a) => `<tr><th>${a}</th>${order.map((b) => {
    if (a === b) return `<td class="muted">—</td>`;
    const v = sv(a, b); return `<td style="background:rgba(81,70,229,${Math.max(0, (v - 0.5) * 1.6).toFixed(2)})">${fmt(v, 2)}</td>`; }).join("")}</tr>`).join("")}</table>`;
  const V = F.vectors || [];
  const feats = (p.features || []).filter((f) => !f.off);
  el.innerHTML = `<div class="row" style="gap:8px;flex-wrap:wrap">
      <span class="seg" id="txsplit">${["train", "validation"].map((s) => `<button data-v="${s}" class="${X.split === s ? "on" : ""}">${s}</button>`).join("")}</span>
      <button class="btn sm" id="txprev">◀</button><input type="number" id="txidx" value="${p.index}" min="0" max="${p.n - 1}" style="width:80px"><span class="small muted">of ${fmtN(p.n)}</span><button class="btn sm" id="txnext">▶</button>
      <button class="btn sm" id="txrand">random</button><span class="sp"></span>
      <input type="text" id="txq" placeholder="find examples with… (e.g. never)" style="width:200px"><button class="btn sm" id="txfind">find</button></div>
    <div id="txhits" class="tiny" style="margin:4px 0"></div><div id="txperr"></div>
    <div class="tx-story">${sent("o1")}${sent("h1")}${sent("h2")}${sent("o2")}</div>
    <div class="tiny muted" style="margin:4px 0 12px">Word chips = what the pipeline produces. <span class="tx-tok unk">red</span> = no GloVe vector (${esc(cfg().text.embedding.unknown)}),
      <span class="tx-tok low">dotted</span> = found only lowercased. Click a sentence to follow it below, a word to see its vector and neighbours.
      ${p.only_h1.length + p.only_h2.length ? `<br>The hypotheses differ in: h1 <b>${esc(p.only_h1.join(" ") || "—")}</b> · h2 <b>${esc(p.only_h2.join(" ") || "—")}</b> (${p.same_words.length} words shared).` : ""}</div>
    <div class="grid g2" style="align-items:start">
      <div class="card"><h3>Step by step: ${esc(FIELD_NAMES[X.field])}</h3><div class="tx-stages">${stages}</div></div>
      <div class="card"><h3>What the model gets</h3>
        <div class="small ink2">${fmtK === "score_each" ? `Pass 1 reads (O1, O2, <b>h1</b>), pass 2 reads (O1, O2, <b>h2</b>), with the same weights.` : `One pass reads O1, O2, h1 and h2.`}
          Every word below is one row of ${p.dim} numbers (first 64 shown); the last row is their mean (what a mean-pooling block makes of the sentence).</div>
        <div class="tx-vecs">${V.map((row, i) => `<div class="tx-vrow"><span class="mono tiny">${esc(F.vector_words[i])}</span>${vstrip(row)}</div>`).join("")}
          ${F.mean ? `<div class="tx-vrow mean"><span class="mono tiny"><b>mean</b></span>${vstrip(F.mean)}</div>` : ""}</div>
        <div class="sect-h">Sentence similarity after averaging (cosine)</div>${simTable}
        <div class="tiny muted" style="margin-top:4px">1 = the mean vectors point the same way. If h1 and h2 are almost 1, an averaging model sees nearly the same input for both:
          the few words that differ get drowned out.</div>
        ${feats.length ? `<div class="sect-h">Own features on this example</div><table class="stbl">${["name", "h1", "h2"].map((c) => `<th>${c}</th>`).join("")}${feats.map((f) => `<tr><td><code>${esc(f.name)}</code></td>${f.error ? `<td colspan="2" class="l">${esc(f.error)}</td>` : `<td>${fmt(f.h1, 3)}</td><td>${fmt(f.h2, 3)}</td>`}</tr>`).join("")}</table>` : ""}
      </div></div>`;
  bindMid();
}

function vstrip(vals) {
  const m = Math.max(0.6, ...vals.map((v) => Math.abs(v)));
  return `<span class="tx-vs">${vals.map((v) => `<i style="background:${v >= 0 ? `rgba(210,59,59,${(Math.min(1, v / m) * 0.9 + 0.05).toFixed(2)})` : `rgba(42,120,214,${(Math.min(1, -v / m) * 0.9 + 0.05).toFixed(2)})`}" title="${fmt(v, 3)}"></i>`).join("")}</span>`;
}

function bindMid() {
  const go = (i) => { X.index = Math.max(0, Math.min((X.prev?.n || 1) - 1, i)); refresh(false); };
  $$("#txsplit button").forEach((b) => b.onclick = () => { X.split = b.dataset.v; X.index = 0; refresh(false); });
  $("#txprev").onclick = () => go(X.index - 1);
  $("#txnext").onclick = () => go(X.index + 1);
  $("#txidx").onchange = (e) => go(+e.target.value);
  $("#txrand").onclick = () => go(Math.floor(Math.random() * (X.prev?.n || 1)));
  const find = async () => {
    const q = $("#txq").value.trim(); if (!q) return;
    const r = await api("/api/text/search", { split: X.split, q });
    $("#txhits").innerHTML = r.indices.length ? `${r.indices.length}${r.indices.length >= 200 ? "+" : ""} examples: ` + r.indices.slice(0, 40).map((i) => `<a href="#" data-go="${i}">#${i}</a>`).join(" ") : "no example contains all of these";
    $$("#txhits [data-go]").forEach((a) => a.onclick = (e) => { e.preventDefault(); go(+a.dataset.go); });
  };
  $("#txfind").onclick = find;
  $("#txq").onkeydown = (e) => { if (e.key === "Enter") find(); };
  $$(".tx-sent").forEach((el) => el.onclick = (e) => { if (e.target.dataset.word) return; X.field = el.dataset.field; renderMid(); });
  $$("#txmid [data-word]").forEach((el) => el.onclick = () => showWord(el.dataset.word));
  updateTextErrors(X.ED.analysis);
}

async function showWord(w) {
  X.word = await api("/api/text/word", { word: w, dim: cfg().text.embedding.dim }).catch((e) => ({ word: w, error: e.message }));
  renderRight();
}

// ---------------------------------------------------------------------------------------------- right: the whole dataset, a word, a feature
function renderRight() {
  const el = $("#txright");
  if (!el) return;
  const s = X.stats;
  let h = `<div class="pal">`;
  if (X.word) {
    const w = X.word;
    h += `<div class="card" style="margin-bottom:10px"><div class="row"><h3 class="mono" style="flex:1">${esc(w.word)}</h3><button class="icb sm" id="txwx">✕</button></div>
      ${w.known ? `<div class="tiny muted">GloVe row ${w.rank}${w.exact ? "" : " (found lowercased)"} · length ${fmt(w.norm, 2)} · ${w.vector.length} numbers:</div>
        <div style="margin:4px 0">${vstrip(w.vector)}</div>
        <div class="sect-h">Nearest words (cosine, top 50k words)</div><div>${w.neighbours.map((n) => `<span class="tx-tok" data-word="${esc(n.word)}">${esc(n.word)} <small>${fmt(n.sim, 2)}</small></span>`).join("")}</div>
        <div class="tiny muted" style="margin-top:6px">Words used in similar contexts get similar vectors. Antonyms often end up close too (they appear in the same places): the vector alone cannot tell "never" from "always".</div>`
        : `<div class="small l">No GloVe vector: it becomes "${esc(cfg().text.embedding.unknown)}". ${esc(X.ED.src.unknown[cfg().text.embedding.unknown] || "")}</div>`}</div>`;
  }
  const F = cfg().text_features;
  if (X.feat != null && F[X.feat]) {
    h += `<div class="card" style="margin-bottom:10px" id="txfeat"><div class="row"><h3 class="mono" style="flex:1">${esc(F[X.feat].name)}</h3>
      <span class="seg" id="txfsplit">${["train", "validation"].map((v) => `<button data-v="${v}" class="${X.featSplit === v ? "on" : ""}">${v}</button>`).join("")}</span></div>
      <div class="tiny muted">${esc(F[X.feat].description || "")}</div><div id="txfbody" class="small muted">computing…</div></div>`;
  }
  h += `<div class="card"><div class="row"><h3 style="flex:1">The whole split</h3><span class="seg" id="txssplit">${["train", "validation"].map((v) => `<button data-v="${v}" class="${X.statsSplit === v ? "on" : ""}">${v}</button>`).join("")}</span></div>`;
  if (!s) h += `<div class="small muted">counting…</div>`;
  else if (s.error) h += `<div class="errbox">${esc(s.error)}</div>`;
  else {
    const L = s.lengths;
    h += `<div class="minis" style="grid-template-columns:1fr 1fr">
        <div class="mini"><div class="k">words</div><div class="v">${fmtN(s.tokens)}</div><div class="s">${fmtN(s.types)} different</div></div>
        <div class="mini"><div class="k">without vector</div><div class="v ${s.unknown_token_share > 0.02 ? "l" : ""}">${fmt(s.unknown_token_share * 100, 2)} %</div><div class="s">${fmtN(s.unknown_types)} different words</div></div>
        <div class="mini"><div class="k">h1 vs h2 after mean</div><div class="v">${fmt(s.h1_h2_cosine_mean, 2)}</div><div class="s">average cosine</div></div>
        <div class="mini"><div class="k">pairs differing by ≤ 4 words</div><div class="v">${Math.round(s.pairs_differing_by_4_words_or_less * 100)} %</div></div></div>
      <div class="sect-h">Words per sentence</div><table class="stbl"><tr><th></th><th>mean</th><th>median</th><th>95 %</th><th>max</th><th>empty</th></tr>
        ${["o1", "o2", "h1", "h2"].map((f) => `<tr><td>${f}</td><td>${fmt(L[f].mean, 1)}</td><td>${L[f].p50}</td><td>${L[f].p95}</td><td>${L[f].max}</td><td class="${L[f].empty ? "l" : ""}">${L[f].empty}</td></tr>`).join("")}</table>
      <div class="sect-h">Most frequent words without a vector</div><div>${s.top_unknown.length ? s.top_unknown.map(([w, c]) => `<span class="tx-tok unk" data-find="${esc(w)}" title="click: find examples">${esc(w)} <small>${c}</small></span>`).join("") : `<span class="small muted">none</span>`}</div>
      <div class="sect-h">Most frequent words</div><div>${s.top_words.slice(0, 20).map(([w, c]) => `<span class="tx-tok" data-word="${esc(w)}">${esc(w)} <small>${fmtN(c)}</small></span>`).join("")}</div>
      <div class="tiny muted" style="margin-top:8px">A mean of word vectors is dominated by the most frequent words. Try "Remove stopwords" (keeping not / never / before / after) and watch h1 vs h2.</div>`;
  }
  h += `</div></div>`;
  el.innerHTML = h;
  if ($("#txwx")) $("#txwx").onclick = () => { X.word = null; renderRight(); };
  $$("#txright [data-word]").forEach((x) => x.onclick = () => showWord(x.dataset.word));
  $$("#txright [data-find]").forEach((x) => x.onclick = () => { $("#txq").value = x.dataset.find; $("#txfind").click(); });
  $$("#txssplit button").forEach((b) => b.onclick = async () => { X.statsSplit = b.dataset.v; X.stats = null; renderRight(); X.stats = await api("/api/text/stats", { inputs: cfg(), split: X.statsSplit }).catch((e) => ({ error: e.message })); renderRight(); });
  $$("#txfsplit button").forEach((b) => b.onclick = () => { X.featSplit = b.dataset.v; renderRight(); });
  if ($("#txfbody")) featureCheck();
}

async function featureCheck() {
  const f = cfg().text_features[X.feat];
  const box = $("#txfbody");
  try {
    const r = await api("/api/text/feature", { inputs: cfg(), formula: f.formula, split: X.featSplit });
    if (!$("#txfbody")) return;
    const best = Math.max(r.acc_higher, r.acc_lower), dir = r.acc_higher >= r.acc_lower ? "higher" : "lower";
    const H = Math.max(1, ...r.hist_plausible, ...r.hist_other);
    const bars = r.hist_plausible.map((a, i) => `<div class="tx-hb"><i class="g" style="height:${(a / H) * 100}%"></i><i class="r" style="height:${(r.hist_other[i] / H) * 100}%"></i></div>`).join("");
    const ex = (e) => `<div class="tx-ex"><div><span class="pill ok">plausible</span> ${esc(e.h_plausible)} <b>${fmt(e.plausible, 2)}</b></div><div><span class="pill bad">other</span> ${esc(e.h_other)} <b>${fmt(e.other, 2)}</b></div>
      <div class="muted">O1: ${esc(e.o1)} · O2: ${esc(e.o2)}</div></div>`;
    box.innerHTML = `<div class="minis" style="grid-template-columns:1fr 1fr;margin-top:6px">
        <div class="mini"><div class="k">alone: pick the ${dir}</div><div class="v ${best >= 0.55 ? "g" : ""}">${fmt(best * 100, 1)} %</div><div class="s">chance = 50 %</div></div>
        <div class="mini"><div class="k">same value for both</div><div class="v">${Math.round(r.ties * 100)} %</div><div class="s">no help there</div></div></div>
      <div class="small" style="margin-top:6px">mean for the plausible hypothesis <b>${fmt(r.mean_plausible, 3)}</b> vs the other <b>${fmt(r.mean_other, 3)}</b> (spread ${fmt(r.std, 3)})</div>
      <div class="tx-hist">${bars}</div><div class="row tiny muted"><span>${fmt(r.bins[0], 2)}</span><span class="sp"></span><span><i class="tx-key g"></i>plausible <i class="tx-key r"></i>other</span><span class="sp"></span><span>${fmt(r.bins[r.bins.length - 1], 2)}</span></div>
      <div class="tiny muted" style="margin-top:6px">"Alone" = a rule that only looks at this number and picks the hypothesis with the ${dir} value (${fmtN(r.n)} ${esc(r.split)} examples).
        It shows whether the number carries signal; the network can still combine it with everything else. Choose features on train or validation, never on test.</div>
      <details class="box" style="margin-top:6px"><summary class="small">where it helps most / misleads most</summary>${r.helps.slice(0, 2).map(ex).join("")}<div class="sect-h">misleads</div>${r.misleads.slice(0, 2).map(ex).join("")}</details>`;
  } catch (e) { if (box) box.innerHTML = `<div class="errbox">${esc(e.message)}</div>`; }
}
