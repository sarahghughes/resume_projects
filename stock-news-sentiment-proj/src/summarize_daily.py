#!/usr/bin/env python3
"""
Summarize sentiment per (ticker, date) and join with same-day open→close return.

Appends/updates rows in outputs/analytics/daily_summary.csv (id = ticker+date).

Usage:
  python src/summarize_daily.py --date 2025-12-11 --tickers @tickers.txt --quiet
"""

import argparse
import csv
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", message="Mean of empty slice")

COLS = [
    "ticker","date","n_headlines","n_pos","n_neg","n_neu",
    "mean_signed","median_signed","mean_score",
    "open","close","open_close_return",
    "pos_rate","neg_rate","neu_rate","frac_pos","frac_neg",
    "pos_count","neg_count","neu_count"
]

def parse_tickers(arg: str) -> list[str]:
    arg = arg.strip()
    if arg.startswith("@"):
        fp = Path(arg[1:])
        return [t.strip().upper() for t in fp.read_text().splitlines() if t.strip() and not t.startswith("#")]
    return [t.strip().upper() for t in arg.split(",") if t.strip()]

def read_scored_csv(repo: Path, tkr: str, date_str: str) -> pd.DataFrame | None:
    yyyy_mm = date_str[:7]
    fp = repo / "data" / "raw" / tkr / yyyy_mm / f"news_{tkr}_{date_str}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    # require scored
    if "fb_label" not in df.columns or "fb_signed" not in df.columns:
        return None
    return df

def summarize_sentiment(df: pd.DataFrame) -> dict | None:
    sub = df.copy()
    # consider only scored rows
    sub = sub[~sub["fb_label"].isna() & (sub["fb_label"].astype(str) != "")]
    if sub.empty:
        return None
    labels = sub["fb_label"].astype(str).str.lower()
    n_total = int(len(sub))
    n_pos = int((labels == "positive").sum())
    n_neg = int((labels == "negative").sum())
    n_neu = int((labels == "neutral").sum())
    mean_signed = float(np.nanmean(sub["fb_signed"].astype(float).to_numpy()))
    median_signed = float(np.nanmedian(sub["fb_signed"].astype(float).to_numpy()))
    mean_score = float(np.nanmean(sub["fb_score"].astype(float).to_numpy()))
    pos_rate = n_pos / n_total if n_total else 0.0
    neg_rate = n_neg / n_total if n_total else 0.0
    neu_rate = n_neu / n_total if n_total else 0.0
    # alias fields kept for convenience
    return {
        "n_headlines": n_total,
        "n_pos": n_pos, "n_neg": n_neg, "n_neu": n_neu,
        "mean_signed": mean_signed, "median_signed": median_signed, "mean_score": mean_score,
        "pos_rate": pos_rate, "neg_rate": neg_rate, "neu_rate": neu_rate,
        "frac_pos": pos_rate, "frac_neg": neg_rate,
        "pos_count": n_pos, "neg_count": n_neg, "neu_count": n_neu,
    }

def fetch_ohlc_open_close(tkr: str, date_str: str) -> tuple[float|None, float|None, float|None]:
    # yfinance: fetch [date, date+1) to get that day’s bar
    d = dt.date.fromisoformat(date_str)
    start = d
    end = d + dt.timedelta(days=1)
    try:
        df = yf.download(tkr, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None, None, None
    if df is None or df.empty:
        return None, None, None
    day = df.iloc[0]
    o = float(day["Open"]); c = float(day["Close"])
    ret = (c / o - 1.0) if (o and o == o) else None
    return o, c, ret

def upsert_summary(repo: Path, rows: list[dict]):
    out_dir = repo / "outputs" / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "daily_summary.csv"
    if out_fp.exists():
        master = pd.read_csv(out_fp)
    else:
        master = pd.DataFrame(columns=COLS)

    # build key and upsert
    key = master["ticker"].astype(str) + "," + master["date"].astype(str) if not master.empty else pd.Series(dtype=str)
    buf = master.copy()
    for r in rows:
        k = f"{r['ticker']},{r['date']}"
        if not buf.empty and (key == k).any():
            buf.loc[(buf["ticker"] == r["ticker"]) & (buf["date"] == r["date"]), COLS] = [r.get(c, "") for c in COLS]
        else:
            buf = pd.concat([buf, pd.DataFrame([[r.get(c, "") for c in COLS]], columns=COLS)], ignore_index=True)
    buf.to_csv(out_fp, index=False)
    return out_fp, len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--tickers", required=True, help="AAPL,MSFT or @tickers.txt")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        args.quiet = False

    repo = Path(__file__).resolve().parents[1]
    tickers = parse_tickers(args.tickers)

    new_rows = []
    for t in tickers:
        df = read_scored_csv(repo, t, args.date)
        if df is None:
            if args.verbose:
                print(f"[SUM] {t} {args.date} – no file or not scored yet")
            continue
        sent = summarize_sentiment(df)
        if sent is None:
            if args.verbose:
                print(f"[SUM] {t} {args.date} – no scored rows; skipping prices")
            continue
        o, c, r = fetch_ohlc_open_close(t, args.date)
        row = {
            "ticker": t, "date": args.date,
            "n_headlines": sent["n_headlines"],
            "n_pos": sent["n_pos"], "n_neg": sent["n_neg"], "n_neu": sent["n_neu"],
            "mean_signed": sent["mean_signed"], "median_signed": sent["median_signed"], "mean_score": sent["mean_score"],
            "open": o if o is not None else "",
            "close": c if c is not None else "",
            "open_close_return": r if r is not None else "",
            "pos_rate": sent["pos_rate"], "neg_rate": sent["neg_rate"], "neu_rate": sent["neu_rate"],
            "frac_pos": sent["frac_pos"], "frac_neg": sent["frac_neg"],
            "pos_count": sent["pos_count"], "neg_count": sent["neg_count"], "neu_count": sent["neu_count"],
        }
        new_rows.append(row)

    if not new_rows:
        if not args.quiet:
            print("=== Coverage ===")
            print("rows (sentiment+return): 0"); print("tickers: —"); print("date range: —")
        return 0

    out_fp, n = upsert_summary(repo, new_rows)
    if not args.quiet:
        tick_list = sorted(set(r["ticker"] for r in new_rows))
        print(f"[SUMMARY] appended/updated {n} rows → {out_fp}")
        print("=== Coverage ===")
        print(f"rows (sentiment+return): {n}")
        print(f"tickers: {', '.join(tick_list)}")
        print(f"date range: {new_rows[0]['date']} → {new_rows[0]['date']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
