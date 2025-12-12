#!/usr/bin/env python3
"""
Score headlines for a UTC date across tickers with FinBERT.
Writes fb_label, fb_score, fb_signed back into each day's CSV.

Usage:
  python src/score_finbert_day.py --date 2025-12-11 --tickers @tickers.txt --device auto
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()  # silence transformers chatter

LABEL_SIGN = {"positive": +1.0, "negative": -1.0, "neutral": 0.0}

def parse_tickers(arg: str) -> list[str]:
    arg = arg.strip()
    if arg.startswith("@"):
        fp = Path(arg[1:])
        return [t.strip().upper() for t in fp.read_text().splitlines() if t.strip() and not t.startswith("#")]
    return [t.strip().upper() for t in arg.split(",") if t.strip()]

def device_arg(s: str) -> str:
    s = (s or "auto").lower()
    if s in {"auto","cpu"}: return s
    if s.startswith("cuda"): return s
    return "auto"

def get_pipe(dev: str, batch_size: int):
    if dev == "auto":
        dev = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    pipe = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=dev,
        return_all_scores=True,
        truncation=True,
        max_length=256
    )
    pipe.batch_size = batch_size
    return pipe

def score_texts(pipe, texts: list[str]) -> tuple[list[str], list[float], list[float]]:
    out_labels, out_scores, out_signed = [], [], []
    for res in pipe(texts):
        # res is list of dicts with labels/probs
        probs = {d["label"].lower(): float(d["score"]) for d in res}
        label = max(probs, key=probs.get)
        score = probs[label]
        signed = score * LABEL_SIGN[label]
        out_labels.append(label)
        out_scores.append(score)
        out_signed.append(signed)
    return out_labels, out_scores, out_signed

def score_file(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    for col in ["fb_label", "fb_score", "fb_signed"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)

    # choose unscored rows
    idx = df.index[(df["fb_label"].isna()) | (df["fb_label"] == "")]
    if len(idx) == 0:
        return 0

    texts = df.loc[idx, "headline"].astype(str).fillna("").tolist()
    labels, scores, signed = score_texts(SCORE_PIPE, texts)

    # assign with correct dtype to kill pandas warnings
    df.loc[idx, "fb_label"]  = pd.Series(labels, index=idx, dtype=object)
    df.loc[idx, "fb_score"]  = pd.Series(scores, index=idx, dtype=object)
    df.loc[idx, "fb_signed"] = pd.Series(signed, index=idx, dtype=object)

    df.to_csv(csv_path, index=False)
    return len(idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--tickers", required=True, help="AAPL,MSFT or @tickers.txt")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tickers = parse_tickers(args.tickers)

    global SCORE_PIPE
    SCORE_PIPE = get_pipe(device_arg(args.device), args.batch_size)

    total = 0
    for t in tickers:
        yyyy_mm = args.date[:7]
        fp = repo / "data" / "raw" / t / yyyy_mm / f"news_{t}_{args.date}.csv"
        n = score_file(fp)
        print(f"[{t}] {args.date}: scored {n} rows in {fp}")
        total += n
    print(f"[SCORE] total newly scored rows: {total}")

if __name__ == "__main__":
    main()
