// Text sources (ART), everything outside the Inputs tab: Task & loss, Training settings, the train dialog, "Look inside" on a sentence,
// and the run page extras (data card, the one-time test, the error browser).
import { $, $$, api, esc, fmt, fmtN, icon, modal, toast } from "./util.js";
import { barList } from "./charts.js";

// ---------------------------------------------------------------------------------------------- Task & loss (instead of Reward)
export function tabTextTask(body, ED) {
  body.style.overflow = "auto";
  const f = ED.d.inputs?.text?.format || "score_each";
  body.innerHTML = `<div class="page" style="max-width:980px"><h3>Task &amp; loss</h3>
    <div class="card"><p class="small ink2" style="line-height:1.6">${esc(ED.src.description)}</p>
      <div class="sect-h">How the model is trained (supervised learning)</div>
      <div class="small ink2" style="line-height:1.7">
        1. A batch of examples goes through the model. ${f === "score_each" ? "Each example twice: once with h1, once with h2 → two scores s1, s2." : "Each example once → two scores (h1 right, h2 right)."}<br>
        2. <b>Softmax</b> turns the two scores into probabilities: p(h2) = e<sup>s2</sup> / (e<sup>s1</sup> + e<sup>s2</sup>). Only the difference s2 − s1 matters.<br>
        3. <b>Cross-entropy loss</b> = −log(probability given to the right hypothesis): 0.69 when the model says 50/50, close to 0 when it is sure and right,
           large when it is sure and wrong.<br>
        4. Backpropagation computes how every trainable weight should change to lower the loss; the optimizer (Adam) takes a small step.
           The word vectors are frozen: only the blocks after them learn.<br>
        5. After every epoch the model is measured on <b>validation</b> (1,000 examples it never trains on). The epoch with the best validation number is kept
           (early stopping). The <b>test</b> split (1,532) is used once, at the very end, with a button on the run page.</div>
      <div class="sect-h">Reading the numbers</div>
      <div class="small ink2" style="line-height:1.7"><b>accuracy</b> = share of examples where the right hypothesis got the higher score (chance 50 %).
        Validation has 1,000 examples, so ±3 % is noise (95 % range): repeat a setup with 2-3 seeds before believing a 1-2 % difference.
        <b>Overfitting</b>: train accuracy keeps rising while validation loss starts to climb — early stopping catches that moment.</div></div></div>`;
}

// ---------------------------------------------------------------------------------------------- Training
export function tabTextTraining(body, ED, markDirty) {
  body.style.overflow = "auto";
  const tr = ED.d.training ||= {};
  const T = ED.src.training;
  const val = (k, d) => tr[k] ?? d;
  body.innerHTML = `<div class="page" style="max-width:980px"><h3>Training</h3><p class="muted small">Runs here in the Studio on the Deck's CPU (own process, ${val("threads", 2)} threads). Saved with the design.</p>
    <div class="card"><div class="grid g2">${T.numbers.map((p) => `<label class="f">${esc(p.label)}<input type="number" data-tk="${p.key}" value="${esc(val(p.key, p.default))}" min="${p.min}" max="${p.max}" step="${p.step}">
        <span class="small muted" style="font-weight:400">${esc(p.help)} <span class="muted">default ${p.default}</span></span></label>`).join("")}
      ${T.choices.map((p) => `<label class="f">${esc(p.label)}<select data-tc="${p.key}">${p.choices.map((c) => `<option ${c === val(p.key, p.default) ? "selected" : ""}>${c}</option>`).join("")}</select>
        <span class="small muted" style="font-weight:400">${esc(p.help)}</span></label>`).join("")}
      <label class="f">CPU threads<input type="number" data-tk="threads" value="${esc(val("threads", 2))}" min="1" max="6" step="1"><span class="small muted" style="font-weight:400">More = faster, but the Deck runs other things too (Civ Lab, bots). 2-4.</span></label></div>
      ${T.bools.map((p) => `<label class="row small" style="margin-top:8px"><input type="checkbox" data-tb="${p.key}" ${val(p.key, p.default) ? "checked" : ""}> <b>${esc(p.label)}</b> <span class="muted">— ${esc(p.help)}</span></label>`).join("")}
      <div class="row" style="margin-top:8px;gap:10px"><label class="f small">W&amp;B project<input type="text" data-ts="wandb_project" value="${esc(val("wandb_project", "art-p1"))}"></label>
        <label class="f small">W&amp;B team<input type="text" data-ts="wandb_entity" value="${esc(val("wandb_entity", "sascha_ai"))}"></label></div></div>
    <div class="learn" style="margin-top:14px"><b>A good order of experiments:</b> start with the defaults; then change <i>one</i> thing per run (pipeline step, pooling, hidden size, dropout, lr) and compare
      on validation in Experiments. When a setup looks best, repeat it with seeds 1 and 2. Only then press "Final test" on the run you picked — once.</div></div>`;
  $$("[data-tk]").forEach((el) => el.oninput = () => { tr[el.dataset.tk] = +el.value; markDirty(); });
  $$("[data-tc]").forEach((el) => el.onchange = () => { tr[el.dataset.tc] = el.value; markDirty(); });
  $$("[data-tb]").forEach((el) => el.onchange = () => { tr[el.dataset.tb] = el.checked; markDirty(); });
  $$("[data-ts]").forEach((el) => el.oninput = () => { tr[el.dataset.ts] = el.value.trim(); markDirty(); });
}

// ---------------------------------------------------------------------------------------------- train dialog
export function textTrainDialog(ED, save) {
  const tr = { epochs: 30, patience: 3, lr: 0.001, batch_size: 64, seed: 0, ...(ED.d.training || {}) };
  const t = ED.d.inputs?.text || {};
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  const def = `${slug(ED.d.name)}-v${ED.d.version + (ED.dirty ? 1 : 0)}-s${tr.seed}`;
  modal(`<h3>Train ${esc(ED.d.name)}</h3><p class="small muted">Runs here in the Studio on the CPU. The design is saved first and the run keeps a snapshot of it.
      ${tr.wandb !== false ? `It is mirrored to W&amp;B (${esc(tr.wandb_entity || "sascha_ai")}/${esc(tr.wandb_project || "art-p1")}).` : "W&amp;B logging is off (Training tab)."}</p>
    <label class="f">run name<input type="text" id="tname" value="${esc(def)}"><span class="small muted" style="font-weight:400">Shown in Experiments and W&amp;B. Say what is special: e.g. "a1-mean-stopwords-off-s0".</span></label>
    <div class="row" style="margin-top:8px"><label class="f">epochs<input type="number" id="tep" value="${tr.epochs}" min="1"></label><label class="f">patience<input type="number" id="tpat" value="${tr.patience}" min="0"></label>
      <label class="f">lr<input type="number" id="tlr" value="${tr.lr}" step="0.0001"></label><label class="f">batch<input type="number" id="tbs" value="${tr.batch_size}" min="4"></label>
      <label class="f">seed<input type="number" id="tseed" value="${tr.seed}"></label></div>
    <label class="f" style="margin-top:8px">note (what do you expect?)<input type="text" id="tnote" placeholder="e.g. removing stopwords should make h1 and h2 look more different"></label>
    <div class="small muted" style="margin-top:8px">format: <b>${esc(ED.src.formats[t.format || "score_each"].label)}</b> · GloVe ${esc(t.embedding?.dim || 300)}d · ${(t.steps || []).filter((s) => !s.off).map((s) => esc(s.op)).join(" → ")}</div>
    <div class="row" style="margin-top:14px"><label class="row small"><input type="checkbox" id="tseeds"> also start a second run with seed +1 (2 runs in parallel)</label><span class="sp"></span>
      <button class="btn primary" id="tgo">${icon("play")} Start</button></div>`, (m, close) => {
    $("#tgo", m).onclick = async () => {
      try {
        const vals = { epochs: +$("#tep", m).value, patience: +$("#tpat", m).value, lr: +$("#tlr", m).value, batch_size: +$("#tbs", m).value, seed: +$("#tseed", m).value };
        let ch = false;
        for (const [k, v] of Object.entries(vals)) if ((ED.d.training || {})[k] !== v) { ch = true; }
        if (ch) { ED.d.training = { ...(ED.d.training || {}), ...vals }; ED.dirty = true; }
        if (ED.dirty) await save("before training");
        const name = slug($("#tname", m).value) || def;
        const r = await api("/api/run", { design: ED.d.id, name, seed: vals.seed, note: $("#tnote", m).value });
        if ($("#tseeds", m).checked) {
          await api("/api/run", { design: ED.d.id, name: name.replace(/-s\d+$/, "") + `-s${vals.seed + 1}`, seed: vals.seed + 1, note: $("#tnote", m).value }).catch((e) => toast(e.message, true));
        }
        close(); location.hash = `#/run/${encodeURIComponent(r.id)}`;
      } catch (e) { toast(e.message, true); }
    };
  });
}

// ---------------------------------------------------------------------------------------------- Look inside (inspector)
export function renderTextLook(r, out, ED) {
  const b = r.base, dec = r.decision;
  const fields = Object.keys(b.words);
  let h = `<div class="small muted">${esc(r.note)}</div>${storyHtml(b)}`;
  if (dec) h += decisionHtml(dec, b);
  for (const [port, v] of Object.entries(r.ports)) {
    h += `<div class="sect-h">${esc(port)} · ${v.kind} · ${v.dim} numbers${v.kind === "set" ? ` × ${v.n} items` : ""}</div>`;
    if (v.kind === "set") {
      const words = v.n === (b.words[port]?.length) ? b.words[port] : guessWords(v.n, b, fields);
      h += `<div class="tx-vecs">${v.rows.map((row, i) => `<div class="tx-vrow"><span class="mono tiny">${esc(words?.[i] ?? "#" + i)}</span>${vstrip(row.slice(0, 64))}</div>`).join("")}</div>
        <div class="tiny muted">first 64 of ${v.dim} numbers per item · mean ${fmt(v.stats.mean, 3)}, ${Math.round(v.stats.zero * 100)} % zero</div>`;
    } else {
      h += `<div>${vstrip(v.values.slice(0, 256))}</div><div class="tiny muted">${v.dim} numbers · mean ${fmt(v.stats.mean, 3)} · min ${fmt(v.stats.min, 2)} · max ${fmt(v.stats.max, 2)} · ${Math.round(v.stats.zero * 100)} % zero</div>`;
    }
  }
  out.innerHTML = h;
}

function guessWords(n, b, fields) {
  // a joined sequence (O1 h O2 with markers) — rebuild the labels in reading order when the count fits
  const seqs = [["o1", "h", "o2"], ["o1", "h1", "h2", "o2"], ["o1", "h1", "o2"]];
  for (const s of seqs) {
    if (!s.every((f) => b.words[f])) continue;
    const plain = s.flatMap((f) => b.words[f]);
    if (plain.length === n) return plain;
    const marked = s.flatMap((f, i) => (i < s.length - 1 ? [...b.words[f], "·"] : b.words[f]));
    if (marked.length === n) return marked;
  }
  return null;
}

export function storyHtml(b) {
  const lab = b.label;
  const line = (f, t) => {
    const good = (f === "h1" && lab === 0) || (f === "h2" && lab === 1);
    const cur = b.candidate != null && ((f === "h1" && b.candidate === 0) || (f === "h2" && b.candidate === 1));
    return `<div class="tx-line ${cur ? "cur" : ""}"><b>${f.toUpperCase()}</b> ${esc(t)}${good ? ` <span class="pill ok">plausible</span>` : ""}${cur ? ` <span class="pill run">scored now</span>` : ""}</div>`;
  };
  return `<div class="tx-story-s">${["o1", "h1", "h2", "o2"].map((f) => line(f, b.text[f])).join("")}</div>`;
}

export function decisionHtml(dec, b) {
  if (dec.score != null) {
    const right = dec.candidate === dec.label;
    return `<div class="banner ${(dec.p_this >= 0.5) === right ? "" : "warn"}" style="margin:6px 0"><span>score of h${dec.candidate + 1} = <b>${fmt(dec.score, 3)}</b>, of h${2 - dec.candidate} = ${fmt(dec.other_score, 3)} →
      softmax: <b>${Math.round(dec.p_this * 100)} %</b> for h${dec.candidate + 1}. ${(dec.p_this >= 0.5) === right ? "Right" : "Wrong"} (h${dec.label + 1} is plausible).</span></div>`;
  }
  const ok = (dec.probs[1] > dec.probs[0] ? 1 : 0) === dec.label;
  return `<div class="banner ${ok ? "" : "warn"}" style="margin:6px 0"><span>h1 <b>${Math.round(dec.probs[0] * 100)} %</b> · h2 <b>${Math.round(dec.probs[1] * 100)} %</b> — ${ok ? "right" : "wrong"} (h${dec.label + 1} is plausible)</span></div>`;
}

export function vstrip(vals) {
  const m = Math.max(0.6, ...vals.map((v) => Math.abs(v)));
  return `<span class="tx-vs">${vals.map((v) => `<i style="background:${v >= 0 ? `rgba(210,59,59,${(Math.min(1, v / m) * 0.9 + 0.05).toFixed(2)})` : `rgba(42,120,214,${(Math.min(1, -v / m) * 0.9 + 0.05).toFixed(2)})`}" title="${fmt(v, 3)}"></i>`).join("")}</span>`;
}

// ---------------------------------------------------------------------------------------------- run page extras
export function textRunExtras(d, id, host) {
  const st = d.status || {}, data = d.data || st.data;
  const test = st.test;
  let h = `<div class="grid g2" style="margin-bottom:14px">`;
  h += `<div class="card"><h3>Result</h3><div class="minis" style="grid-template-columns:repeat(3,1fr)">
      <div class="mini"><div class="k">best val accuracy</div><div class="v">${st.best ? fmt(st.best.val_acc * 100, 1) + " %" : "—"}</div><div class="s">epoch ${st.best?.epoch ?? "—"} · loss ${fmt(st.best?.val_loss, 3)}</div></div>
      <div class="mini"><div class="k">train accuracy (that epoch)</div><div class="v">${st.best ? fmt(st.best.train_acc * 100, 1) + " %" : "—"}</div><div class="s">gap ${st.best ? fmt((st.best.train_acc - st.best.val_acc) * 100, 1) + " pts" : "—"}</div></div>
      <div class="mini"><div class="k">TEST accuracy</div><div class="v">${test ? fmt(test.acc * 100, 1) + " %" : "—"}</div><div class="s">${test ? `once, ${esc(test.at)}` : "not used yet"}</div></div></div>
    <div class="small muted" style="margin-top:6px">${st.stop_reason ? `Stopped: ${esc(st.stop_reason)} after ${st.epochs_run} epoch(s). ` : ""}${fmtN(d.params)} trainable weights.
      ${st.wandb_url ? `<a href="${esc(st.wandb_url)}" target="_blank">W&amp;B run ↗</a>` : st.wandb_error ? `<span class="l">W&amp;B: ${esc(st.wandb_error)}</span>` : ""}</div>
    ${d.state === "done" || d.state === "stopped" ? `<div class="row" style="margin-top:10px"><button class="btn ${test ? "" : "primary"}" id="rtest">${test ? "Test again…" : "Final test (once)"}</button>
      <span class="small muted">${test ? "Already tested. Testing again and keeping the better number would be tuning on the test set." : "Uses the best epoch's weights on the 1,532 test examples. Do this only for the run you picked on validation."}</span></div>` : ""}</div>`;
  if (data) h += `<div class="card"><h3>Data of this run</h3><div class="kvs">
      <div class="k">vocabulary</div><div class="v">${fmtN(data.vocabulary)} words with a vector, ${fmtN(data.unknown_types)} without</div>
      <div class="k">unknown words</div><div class="v">${fmt(data.unknown_token_share * 100, 2)} % of all words in train</div>
      <div class="k">examples</div><div class="v">${fmtN(st.train_examples_used || data.train_examples)} train (of ${fmtN(data.train_examples)}) · ${fmtN(data.validation_examples)} validation</div>
      <div class="k">own features</div><div class="v">${data.features}</div><div class="k">batches / epoch</div><div class="v">${st.steps_per_epoch ?? "—"}</div></div></div>`;
  h += `</div><div class="card" style="margin-bottom:14px" id="rex"><div class="row"><h3 style="flex:1">Examples (best epoch)</h3>
      <span class="seg" id="rexf">${[["wrong", "wrong, most sure first"], ["unsure", "unsure"], ["right", "right, most sure first"]].map(([k, l], i) => `<button data-v="${k}" class="${i === 0 ? "on" : ""}">${l}</button>`).join("")}</span>
      ${test ? `<span class="seg" id="rexs"><button data-v="validation" class="on">validation</button><button data-v="test">test</button></span>` : ""}</div><div id="rexb" class="small muted">loading…</div></div>`;
  host.innerHTML = h;
  if ($("#rtest")) $("#rtest").onclick = () => {
    const again = !!test;
    if (again && !confirm("This run was already tested. Test again anyway (the old result stays in the log)?")) return;
    if (!again && !confirm("Use the test set now for this run? Do it once, for the run you chose on validation.")) return;
    $("#rtest").disabled = true; $("#rtest").textContent = "testing…";
    api(`/api/run/${encodeURIComponent(id)}/test`, { force: again }).then((r) => { toast(`test accuracy ${fmt(r.acc * 100, 1)} %`); location.reload(); })
      .catch((e) => { toast(e.message, true); $("#rtest").disabled = false; $("#rtest").textContent = "Final test (once)"; });
  };
  const S = { filter: "wrong", split: "validation", offset: 0 };
  const load = async () => {
    try {
      const r = await api(`/api/run/${encodeURIComponent(id)}/examples`, { filter: S.filter, split: S.split, offset: S.offset, limit: 12 });
      const rows = r.items.map((e) => {
        const pr = e.label === 1 ? e.p_h2 : 1 - e.p_h2;
        return `<div class="tx-ex"><div class="row"><span class="tiny muted">${esc(r.split)} #${e.index}</span><span class="sp"></span><b class="${pr >= 0.5 ? "g" : "l"}">${Math.round(pr * 100)} % on the plausible one</b></div>
          <div><b>O1</b> ${esc(e.o1)}</div><div><b>h1</b> ${esc(e.h1)} ${e.label === 0 ? `<span class="pill ok">plausible</span>` : ""} <span class="tiny muted">${Math.round((1 - e.p_h2) * 100)} %</span></div>
          <div><b>h2</b> ${esc(e.h2)} ${e.label === 1 ? `<span class="pill ok">plausible</span>` : ""} <span class="tiny muted">${Math.round(e.p_h2 * 100)} %</span></div><div><b>O2</b> ${esc(e.o2)}</div></div>`;
      }).join("");
      $("#rexb").innerHTML = `<div class="grid g2" style="margin-bottom:10px"><div><div class="sect-h" style="margin-top:0">Accuracy by how different h1 and h2 are</div><div id="rexg"></div></div>
          <div><div class="sect-h" style="margin-top:0">Is the model as sure as it is right? (calibration)</div><div id="rexc"></div></div></div>
        <div class="small muted">${r.total} ${esc(S.filter)} examples on ${esc(r.split)} (accuracy ${fmt(r.acc * 100, 1)} %)</div>${rows}
        <div class="row" style="margin-top:8px">${S.offset > 0 ? `<button class="btn sm" id="rexp">◀</button>` : ""}<span class="small muted">${S.offset + 1}–${Math.min(r.total, S.offset + 12)}</span>${S.offset + 12 < r.total ? `<button class="btn sm" id="rexn">▶</button>` : ""}</div>`;
      barList($("#rexg"), r.groups.map((g) => ({ label: `${g.group} (${g.n})`, value: g.acc * 100, color: "var(--k-set)" })), { fmt: (v) => v.toFixed(1) + " %" });
      barList($("#rexc"), r.confidence.map((g) => ({ label: `sure ${g.confidence} (${g.n})`, value: g.acc * 100, color: "var(--k-head)" })), { fmt: (v) => v.toFixed(1) + " % right" });
      if ($("#rexp")) $("#rexp").onclick = () => { S.offset = Math.max(0, S.offset - 12); load(); };
      if ($("#rexn")) $("#rexn").onclick = () => { S.offset += 12; load(); };
    } catch (e) { $("#rexb").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
  };
  $$("#rexf button").forEach((b) => b.onclick = () => { S.filter = b.dataset.v; S.offset = 0; $$("#rexf button").forEach((x) => x.classList.toggle("on", x === b)); load(); });
  $$("#rexs button").forEach((b) => b.onclick = () => { S.split = b.dataset.v; S.offset = 0; $$("#rexs button").forEach((x) => x.classList.toggle("on", x === b)); load(); });
  load();
}
