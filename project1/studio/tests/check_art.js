// Headless smoke test of the ART (text) pages: Inputs (text), Model + Look inside, Flow, Training, Task, run page, source page.
// node tests/check_art.js [design-id] [run-id] [port]   screenshots in /tmp/xs-art
const { chromium } = require(process.env.PW || 'playwright');
(async () => {
  const [design = 'art-arch-1-dev-test', run = 'zz-art-arch1', port = '8781'] = process.argv.slice(2);
  require('fs').mkdirSync('/tmp/xs-art', { recursive: true });
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  const errs = [];
  p.on('pageerror', e => errs.push('pageerror ' + String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push('console ' + m.text()); });
  p.on('dialog', d => d.accept());
  const U = `http://127.0.0.1:${port}/`;
  const shot = async (name, ms = 1500) => { await p.waitForTimeout(ms); await p.screenshot({ path: `/tmp/xs-art/${name}.png` }); };
  await p.goto(U + `#/design/${design}/inputs`); await shot('inputs', 4000);
  // switch a step off/on, pick a word, add a feature idea, follow h2
  const sw = await p.$$('[data-son]'); if (sw.length > 4) { await sw[4].click(); await p.waitForTimeout(3000); }
  await shot('inputs-stopwords');
  const tok = await p.$('#txmid [data-word]'); if (tok) { await tok.click(); await p.waitForTimeout(1500); }
  await p.selectOption('#txidea', '1'); await shot('inputs-feature', 3500);
  await p.click('.tx-sent[data-field="h2"]'); await shot('inputs-h2', 800);
  await p.goto(U + `#/design/${design}/model`); await p.waitForTimeout(2500);
  const node = await p.$('.gnode[data-id="join"] .hd');
  if (node) { await node.click(); await p.waitForTimeout(800); const r = await p.$('#lirun'); if (r) { await r.click(); await p.waitForTimeout(2500); } }
  await shot('model-look');
  await p.goto(U + `#/design/${design}/flow`); await shot('flow-1', 3000);
  for (let k = 0; k < 7; k++) { const n = await p.$('#flnext'); if (!n) break; await n.click(); await p.waitForTimeout(1500); await p.screenshot({ path: `/tmp/xs-art/flow-${k + 2}.png` }); }
  await p.goto(U + `#/design/${design}/training`); await shot('training');
  await p.goto(U + `#/design/${design}/reward`); await shot('task');
  await p.goto(U + `#/design/${design}/code`); await shot('code');
  await p.goto(U + `#/run/run%3A${run}`); await shot('run', 3500);
  await p.goto(U + '#/source/art'); await shot('source', 2500);
  await p.goto(U + '#/learn'); await shot('learn');
  await p.goto(U + '#/experiments'); await shot('experiments');
  console.log(JSON.stringify(errs, null, 1));
  await b.close();
})();
