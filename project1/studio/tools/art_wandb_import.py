#!/usr/bin/env python3
"""Mirror finished ART runs that were trained without W&B (e.g. on the tower GPU) into W&B afterwards.

    art_wandb_import.py <run> [<run> ...]      runs = folders in data/runs (config.json, status.json, metrics.jsonl)

Logs the same things a live run logs: the config (wandb_config), train/val loss + acc per epoch, best/* and stop_reason in the summary.
A run that already has a wandb_id in its status is skipped.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xcore.train_supervised import wandb_config  # noqa: E402


def mirror(rid):
    import os
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    d = ROOT / "data" / "runs" / rid
    cfg = json.loads((d / "config.json").read_text())
    st = json.loads((d / "status.json").read_text())
    if st.get("wandb_id"):
        return f"{rid}: already on W&B ({st['wandb_id']})"
    design = cfg["design"]
    tr = design.get("training", {})
    lines = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
    name = cfg.get("name") or rid
    config = {**wandb_config(design), "setting": re.sub(r"-s\d+$", "", name), "stage": name.split("-")[0]}   # setting = run without its seed
    run = wandb.init(entity=tr.get("wandb_entity") or "sascha_ai", project=tr.get("wandb_project") or "art-p1", dir=str(d),
                     name=name, group=design.get("id"), config=config, notes=cfg.get("note") or None,
                     tags=[t for t in (design.get("template"), "studio", "tower-gpu") if t])
    data = next((x for x in lines if x.get("phase") == "data"), None)
    if data:
        run.log({"data/" + k: v for k, v in data.items() if k not in ("phase", "epoch", "t")}, step=0)
    train = {x["epoch"]: x for x in lines if x.get("phase") == "train"}
    for v in (x for x in lines if x.get("phase") == "val"):
        e = v["epoch"]
        run.log({"epoch": e, "train/loss": train[e]["loss"], "train/acc": train[e]["acc"], "val/loss": v["loss"], "val/acc": v["acc"]}, step=e)
    run.summary["params"] = st.get("params")
    for k, v in (st.get("best") or {}).items():
        run.summary[f"best/{k}"] = v
    run.summary["stop_reason"] = st.get("stop_reason")
    url, wid = run.url, run.id
    run.finish()
    st.update(wandb_url=url, wandb_id=wid)
    (d / "status.json").write_text(json.dumps(st, indent=1))
    return f"{rid}: {url}"


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(mirror(r))
