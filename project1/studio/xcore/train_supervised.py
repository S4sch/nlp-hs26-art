#!/usr/bin/env python3
"""The Studio's supervised trainer for dataset sources (ART). One run = one directory, like the PPO trainer:

data/runs/<id>/config.json        design snapshot (source, graph, inputs = preprocessing + features, training settings)
              status.json        state, progress, best epoch, final numbers, test result (once), W&B link
              metrics.jsonl      "data" (vocabulary, unknown words), "step" (loss every few batches), "train" / "val" per epoch, "test"
              ckpt/best.pt       weights of the best epoch (by the monitored validation number), ckpt/last.pt the newest
              val_predictions.json   probabilities of the best epoch on every validation example (for the error browser)

    train_supervised.py <run> [threads]     train (stop: touch STOP)
    train_supervised.py <run> --test        evaluate the best weights on the TEST split, once (refused if done before, unless --force)

Rules built in: the test split is never touched while training; hyper-parameters are chosen on validation.
"""
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xcore import text as T  # noqa: E402
from xcore.sources import get as get_source  # noqa: E402
from xcore.sources.art import ArtModel, Encoded, default_training  # noqa: E402


def now():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def flat(o, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
        for i, v in enumerate(o):
            flat(v, f"{prefix}.{i}", out)
    else:
        out[prefix] = o
    return out


def wandb_config(design):
    """What W&B shows as the run's config: the decisions, in readable form."""
    inp = design.get("inputs", {})
    t = T.text_cfg(inp)
    steps = [s["op"] + (f"({s.get('mode')})" if s["op"] == "tokenize" else "") for s in t["steps"] if not s.get("off")]
    cfg = {"design": design.get("name"), "design_id": design.get("id"), "design_version": design.get("version"), "template": design.get("template"),
           "format": t["format"], "pipeline": " > ".join(steps), "embedding": f"{t['embedding']['name']}-{t['embedding']['dim']}d",
           "unknown_words": t["embedding"]["unknown"], "fallback_lower": t["embedding"]["fallback_lower"],
           "features": {f["name"]: f["formula"] for f in T.feature_list(inp)}}
    for n in design["graph"]["nodes"]:
        if n["op"] != "input":
            cfg[f"block.{n['id']}"] = n["op"] + (" " + json.dumps(n.get("params", {}), sort_keys=True) if n.get("params") else "")
    cfg["training"] = {k: v for k, v in design.get("training", {}).items() if k not in ("wandb_entity",)}
    return flat(cfg)


class Run:
    def __init__(self, rid):
        self.rid = rid
        self.dir = ROOT / "data" / "runs" / rid
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.design = self.cfg["design"]
        self.src = get_source(self.design["source"])
        self.tr = {**default_training(), **self.design.get("training", {})}
        self.t0 = time.time()
        (self.dir / "ckpt").mkdir(exist_ok=True)

    def status(self, **kw):
        p = self.dir / "status.json"
        st = json.loads(p.read_text()) if p.exists() else {}
        st.update(kw, updated=now())
        if "minutes" not in kw and st.get("state") == "training":
            st["minutes"] = round((time.time() - self.t0) / 60, 2)
        p.write_text(json.dumps(st, indent=1))
        return st

    def read_status(self):
        p = self.dir / "status.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def metric(self, **kw):
        kw.setdefault("t", round((time.time() - self.t0) / 60, 3))
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")

    def stopped(self):
        return (self.dir / "STOP").exists()

    def minutes_limit(self):
        try:
            return float(json.loads((self.dir / "minutes.json").read_text())["minutes"])
        except (OSError, ValueError, KeyError):
            return float(self.tr["minutes"])

    # ------------------------------------------------------------------ W&B (optional, never fatal)
    def wandb_start(self, resume_id=None):
        if not self.tr.get("wandb"):
            return None
        try:
            import os
            os.environ.setdefault("WANDB_SILENT", "true")
            import wandb
            kw = dict(entity=self.tr.get("wandb_entity") or "sascha_ai", project=self.tr.get("wandb_project") or "art-p1", dir=str(self.dir))
            if resume_id:
                return wandb.init(id=resume_id, resume="allow", **kw)
            return wandb.init(name=self.cfg.get("name") or self.rid, group=self.design.get("id"), config=wandb_config(self.design),
                              tags=[t for t in (self.design.get("template"), T.text_cfg(self.design.get("inputs", {}))["format"], "studio") if t],
                              notes=self.cfg.get("note") or None, **kw)
        except Exception as e:  # noqa: BLE001
            self.status(wandb_error=f"{type(e).__name__}: {e}"[:300])
            return None

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model, enc, split, bs=256):
        model.eval()
        n = enc.n(split)
        losses, probs = [], []
        for a in range(0, n, bs):
            idx = np.arange(a, min(n, a + bs))
            b = enc.batch(split, idx)
            lg = model.logits(b)
            losses.append(F.cross_entropy(lg, b["y"], reduction="sum").item())
            probs.append(torch.softmax(lg, -1)[:, 1].numpy())
        p2 = np.concatenate(probs)
        y = enc.labels[split]
        pred = (p2 > 0.5).astype(np.int64)
        return {"loss": round(sum(losses) / n, 5), "acc": round(float((pred == y).mean()), 5), "n": int(n)}, p2

    # ------------------------------------------------------------------ training
    def run(self):
        tr = self.tr
        seed = int(tr["seed"])
        torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)
        self.status(state="training", started=now(), progress=0.0, message="preparing the data (tokenizing, looking up word vectors)")
        inp = self.design.get("inputs", {})
        enc = Encoded(inp, splits=("train", "validation"))
        toks = [t for r in enc.tok["train"] for f in T.FIELDS for t in r[f]]
        unk_tok = sum(1 for t in toks if t in enc.unknown)
        data = {"vocabulary": len(enc.vocab), "unknown_types": len(enc.unknown), "unknown_token_share": round(unk_tok / max(1, len(toks)), 5),
                "train_examples": enc.n("train"), "validation_examples": enc.n("validation"), "features": enc.n_features}
        self.metric(phase="data", epoch=0, **data)
        model = ArtModel(self.src, self.design, enc.matrix)
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        lr, wd = float(tr["lr"]), float(tr["weight_decay"])
        if tr["optimizer"] == "sgd":
            opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif tr["optimizer"] == "adamw":
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        rng = np.random.default_rng(seed)
        n_train = enc.n("train")
        use = np.sort(rng.permutation(n_train)[:max(16, int(round(n_train * float(tr["train_fraction"]))))])
        bs, epochs = int(tr["batch_size"]), int(tr["epochs"])
        steps_per_epoch = (len(use) + bs - 1) // bs
        monitor = tr["monitor"]
        swap = tr.get("swap_augment") and enc.fmt == "all_in_one"
        wb = self.wandb_start()
        self.status(params=n_params, data=data, message="", wandb_url=getattr(wb, "url", None), wandb_id=getattr(wb, "id", None),
                    train_examples_used=int(len(use)), steps_per_epoch=steps_per_epoch)
        if wb:
            wb.summary["params"] = n_params
            wb.log({"data/" + k: v for k, v in data.items()}, step=0)
        best, best_val, bad, stop_reason, epoch = None, None, 0, "finished all epochs", 0
        log_every = max(1, steps_per_epoch // 20)
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(use)
            tl, tc, tn = 0.0, 0, 0
            for k in range(steps_per_epoch):
                idx = order[k * bs:(k + 1) * bs]
                b = enc.batch("train", idx, swap=(rng.random(len(idx)) < 0.5) if swap else None)
                lg = model.logits(b)
                loss = F.cross_entropy(lg, b["y"])
                opt.zero_grad()
                loss.backward()
                gn = None
                if float(tr["grad_clip"]) > 0:
                    gn = float(torch.nn.utils.clip_grad_norm_(params, float(tr["grad_clip"])))
                opt.step()
                tl += loss.item() * len(idx)
                tc += int((lg.argmax(-1) == b["y"]).sum())
                tn += len(idx)
                if (k + 1) % log_every == 0:
                    line = {"loss": round(loss.item(), 5)}
                    if gn is not None:
                        line["grad_norm"] = round(gn, 4)
                    self.metric(phase="step", epoch=round(epoch - 1 + (k + 1) / steps_per_epoch, 4), **line)
                if self.stopped() or time.time() - self.t0 > self.minutes_limit() * 60:
                    break
            train = {"loss": round(tl / max(1, tn), 5), "acc": round(tc / max(1, tn), 5)}
            val, p2 = self.evaluate(model, enc, "validation")
            self.metric(phase="train", epoch=epoch, **train)
            self.metric(phase="val", epoch=epoch, **{k: v for k, v in val.items() if k != "n"})
            if wb:
                wb.log({"epoch": epoch, "train/loss": train["loss"], "train/acc": train["acc"], "val/loss": val["loss"], "val/acc": val["acc"]}, step=epoch)
            key = val["loss"] if monitor == "val_loss" else -val["acc"]
            if best_val is None or key < best_val - 1e-6:
                best_val, bad = key, 0
                best = {"epoch": epoch, "val_loss": val["loss"], "val_acc": val["acc"], "train_loss": train["loss"], "train_acc": train["acc"]}
                torch.save({"model": model.state_dict(), "epoch": epoch, "design": self.design}, self.dir / "ckpt" / "best.pt")
                (self.dir / "val_predictions.json").write_text(json.dumps({"epoch": epoch, "p_h2": [round(float(x), 6) for x in p2],
                                                                           "label": enc.labels["validation"].tolist()}))
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "epoch": epoch, "design": self.design}, self.dir / "ckpt" / "last.pt")
            self.status(progress=round(epoch / epochs, 3), epoch=epoch, last={"train": train, "val": val}, best=best,
                        final={"score": best["val_acc"], "val_acc": best["val_acc"], "val_loss": best["val_loss"], "epoch": best["epoch"]})
            if self.stopped():
                stop_reason = "stopped by hand"
                break
            if time.time() - self.t0 > self.minutes_limit() * 60:
                stop_reason = f"time limit ({self.minutes_limit():g} min)"
                break
            if int(tr["patience"]) > 0 and bad >= int(tr["patience"]):
                stop_reason = f"early stopping: no better {'validation loss' if monitor == 'val_loss' else 'validation accuracy'} for {bad} epochs"
                break
        if wb:
            for k, v in (best or {}).items():
                wb.summary[f"best/{k}"] = v
            wb.summary["stop_reason"] = stop_reason
            wb.finish()
        self.status(state="stopped" if self.stopped() else "done", finished=now(), progress=1.0, stop_reason=stop_reason, epochs_run=epoch,
                    minutes=round((time.time() - self.t0) / 60, 2))

    # ------------------------------------------------------------------ the one test
    def test(self, force=False):
        st = self.read_status()
        if st.get("test") and not force:
            raise SystemExit(f"this run was already tested on {st['test'].get('at')}; testing again and keeping the better number would be tuning on the test set")
        ck = self.dir / "ckpt" / "best.pt"
        if not ck.exists():
            raise SystemExit("no best checkpoint (the run has not finished an epoch)")
        torch.manual_seed(0)
        enc = Encoded(self.design.get("inputs", {}), splits=("test",))
        model = ArtModel(self.src, self.design, enc.matrix)
        sd = torch.load(str(ck), map_location="cpu")
        model.load_state_dict(sd["model"])
        res, p2 = self.evaluate(model, enc, "test")
        res.update(at=now(), epoch=sd.get("epoch"), forced=bool(force), previous=st.get("test"))
        (self.dir / "test_predictions.json").write_text(json.dumps({"epoch": sd.get("epoch"), "p_h2": [round(float(x), 6) for x in p2],
                                                                    "label": enc.labels["test"].tolist()}))
        self.metric(phase="test", epoch=sd.get("epoch"), loss=res["loss"], acc=res["acc"])
        self.status(test=res)
        wb = self.wandb_start(resume_id=st.get("wandb_id")) if st.get("wandb_id") else None
        if wb:
            wb.summary["test/acc"] = res["acc"]
            wb.summary["test/loss"] = res["loss"]
            wb.summary["test/epoch"] = res["epoch"]
            wb.finish()
        return res


def main():
    rid = sys.argv[1]
    r = Run(rid)
    if "--test" in sys.argv:
        print(json.dumps(r.test(force="--force" in sys.argv)))
        return
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else int(r.tr.get("threads", 2)))
    try:
        open("/proc/self/oom_score_adj", "w").write("900")
    except OSError:
        pass
    try:
        r.run()
    except Exception as e:  # noqa: BLE001
        import traceback
        r.status(state="failed", finished=now(), message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        raise


if __name__ == "__main__":
    main()
