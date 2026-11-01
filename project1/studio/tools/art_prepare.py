#!/usr/bin/env python3
"""Prepare the ART source's data: the course dataset splits and the GloVe vectors.

    tools/art_prepare.py            download what is missing, convert GloVe to memory-mapped numpy files

Files (DATA = xstudio/data/datasets/art, may be a symlink):
  train.jsonl / validation.jsonl / test.jsonl   rows of amarfurt/art-nlp-hs26 (observation_1/2, hypothesis_1/2, label 1|2)
  glove/glove.6B.zip                            the original download (Stanford, 822 MB)
  glove/6B.<dim>.npy                            float32 [400000, dim], same row order as the original (most frequent words first)
  glove/6B.vocab.txt                            one word per line (row i of every .npy)

The Studio memory-maps the .npy files, so only the rows of words that are really looked up are read from disk.
"""
import json
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets" / "art"
HF = "https://huggingface.co/datasets/amarfurt/art-nlp-hs26/resolve/main/anli/{split}-00000-of-00001.parquet"
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
DIMS = (50, 100, 200, 300)


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def dataset():
    for split in ("train", "validation", "test"):
        out = DATA / f"{split}.jsonl"
        if out.exists():
            continue
        import pyarrow.parquet as pq  # only needed once (the ART venv has it)
        pqf = DATA / f"{split}.parquet"
        if not pqf.exists():
            fetch(HF.format(split=split), pqf)
        rows = pq.read_table(pqf).to_pylist()
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} rows", flush=True)


def glove():
    g = DATA / "glove"
    if all((g / f"6B.{d}.npy").exists() for d in DIMS) and (g / "6B.vocab.txt").exists():
        return
    z = g / "glove.6B.zip"
    if not z.exists():
        fetch(GLOVE_URL, z)
    with zipfile.ZipFile(z) as zf:
        for d in DIMS:
            out = g / f"6B.{d}.npy"
            if out.exists():
                continue
            words, rows = [], []
            with zf.open(f"glove.6B.{d}d.txt") as f:
                for line in f:
                    parts = line.decode("utf-8").rstrip("\n").split(" ")
                    words.append(parts[0])
                    rows.append(np.asarray(parts[1:], dtype=np.float32))
            np.save(out, np.stack(rows))
            vf = g / "6B.vocab.txt"
            if not vf.exists():
                vf.write_text("\n".join(words) + "\n")
            print(f"glove {d}d: {len(words)} words", flush=True)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    dataset()
    if "--no-glove" not in sys.argv:
        glove()
    print("ok")
