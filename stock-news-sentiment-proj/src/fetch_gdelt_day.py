#!/usr/bin/env python3
"""
Fetch GDELT headlines for one UTC day across one or many tickers, normalize,
filter, de-duplicate, and save to data/raw/<TICKER>/YYYY-MM/news_<TICKER>_<DATE>.csv

Usage:
  python src/fetch_gdelt_day.py --date 2025-12-11 --tickers AAPL,MSFT
  python src/fetch_gdelt_day.py --date 2025-12-11 --tickers @tickers.txt
"""

import argparse
import csv
import datetime as dt
import re
import sys
import unicodedata
from pathlib import Path

import requests
from dateutil import tz

API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Obvious aggregators/scrapers to cut dupes/noise early
AGGREGATOR_DOMAINS = {
    "biztoc.com", "msn.com", "news.google.com",
    "markets.financialcontent.com", "investingchannel.com",
}

COMPANY_TERMS = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon"],
    "GOOGL": ["google", "alphabet"],
    "META": ["meta", "facebook"],
    "TSLA": ["tesla"],
}

def parse_tickers(arg: str) -> list[str]:
    arg = arg.strip()
    if arg.startswith("@"):
        fp = Path(arg[1:])
        return [t.strip().upper() for t in fp.read_text().splitlines() if t.strip() and not t.startswith("#")]
    return [t.strip().upper() for t in arg.split(",") if t.strip()]

def _canon_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip().lower()
    u = u.split("?")[0].split("#")[0]
    u = u.replace("://m.", "://").replace("://mobile.", "://")
    return u

def _canon_headline(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w$:+()\-]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def is_aggregator(domain: str) -> bool:
    if not domain:
        return False
    d = domain.strip().lower()
    return d in AGGREGATOR_DOMAINS

def looks_relevant(ticker: str, headline: str, url: str, source: str) -> bool:
    t = ticker.upper()
    hay = " ".join(filter(None, [headline, url, source])).lower()

    tick_pats = [
        rf"\b{re.escape(t)}\b",
        rf"\${re.escape(t)}\b",
        rf"\bnasdaq[:\-\s]{re.escape(t)}\b",
        rf"\({re.escape(t)}\)",
    ]
    if any(re.search(p, hay, flags=re.IGNORECASE) for p in tick_pats):
        return True
    for k in COMPANY_TERMS.get(t, []):
        if re.search(rf"\b{re.escape(k)}\b", hay, flags=re.IGNORECASE):
            return True
    return False

def fetch_day(date_str: str, ticker: str, max_items: int = 250):
    d = dt.date.fromisoformat(date_str)
    start = dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz.UTC)
    end   = dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=tz.UTC)
    params = {
        "query": ticker,
        "mode": "ArtList",
        "maxrecords": max_items,
        "format": "JSON",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", []) or []

def norm_row(ticker: str, art: dict, date_str: str) -> dict:
    headline = art.get("title") or art.get("seendate") or ""
    url = art.get("url") or ""
    resolved_url = _canon_url(url)
    pub_date = art.get("seendate") or ""
    pub_date_iso = (art.get("seendate") or "").replace(" ", "T")
    source = (art.get("sourceCommonName") or art.get("domain") or "").strip().lower()
    return {
        "ticker": ticker,
        "headline": headline,
        "url": url,
        "resolved_url": resolved_url,
        "link_kind": "gdelt_docapi",
        "pub_date": pub_date,
        "pub_date_iso": pub_date_iso,
        "rss_url": "",
        "source": source,
        "date": date_str,
        "fb_label": "",
        "fb_score": "",
        "fb_signed": "",
    }

def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen_title, seen_url, out = set(), set(), []
    for r in rows:
        kt = _canon_headline(r.get("headline", ""))
        ku = _canon_url(r.get("resolved_url") or r.get("url") or "")
        if (kt and kt in seen_title) or (ku and ku in seen_url):
            continue
        if kt: seen_title.add(kt)
        if ku: seen_url.add(ku)
        out.append(r)
    return out

def save_csv(rows: list[dict], ticker: str, date_str: str, repo: Path) -> Path:
    d = dt.date.fromisoformat(date_str)
    out_dir = repo / "data" / "raw" / ticker / f"{d.year:04d}-{d.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"news_{ticker}_{date_str}.csv"
    cols = ["ticker","headline","url","resolved_url","link_kind",
            "pub_date","pub_date_iso","rss_url","source","date",
            "fb_label","fb_score","fb_signed"]
    with out_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    return out_fp

def run_for_ticker(date: str, ticker: str, repo: Path, max_items: int):
    arts = fetch_day(date, ticker, max_items=max_items)
    rows = [norm_row(ticker, a, date) for a in arts]
    rows = [r for r in rows if not is_aggregator(r.get("source", ""))]
    rows = [r for r in rows if looks_relevant(ticker, r["headline"], r["resolved_url"] or r["url"], r["source"])]
    rows = dedupe_rows(rows)
    fp = save_csv(rows, ticker, date, repo)
    print(f"[{ticker}] {date}: {len(rows)} headlines → {fp}")

def main():
    ap = argparse.ArgumentParser(description="Fetch GDELT headlines for one UTC day for one/many tickers.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC day)")
    ap.add_argument("--tickers", required=True, help="AAPL,MSFT or @tickers.txt")
    ap.add_argument("--max-items", type=int, default=250)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tickers = parse_tickers(args.tickers)
    for t in tickers:
        try:
            run_for_ticker(args.date, t, repo, args.max_items)
        except Exception as e:
            print(f"[ERROR] {t} {args.date}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
