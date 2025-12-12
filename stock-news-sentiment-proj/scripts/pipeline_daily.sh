#!/usr/bin/env bash
# Orchestrate: fetch → score → summarize for ONE UTC date across tickers.txt

set -euo pipefail

# Quiet most library chatter
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONWARNINGS="ignore:Mean of empty slice,ignore:YF.download() has changed argument auto_adjust default"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="${1:-}"
if [[ -z "${DATE}" ]]; then
  echo "usage: $0 YYYY-MM-DD"
  exit 1
fi

TICKERS_FILE="${REPO}/tickers.txt"
if [[ ! -s "${TICKERS_FILE}" ]]; then
  echo "Missing ${TICKERS_FILE}"; exit 1
fi

# Auto-pick a Python
PY="$(command -v python3 || command -v python)"

echo "=== PIPELINE (UTC date: ${DATE}) ==="

"$PY" "${REPO}/src/fetch_gdelt_day.py"   --date "${DATE}" --tickers "@${TICKERS_FILE}"
"$PY" "${REPO}/src/score_finbert_day.py" --date "${DATE}" --tickers "@${TICKERS_FILE}" --device auto
"$PY" "${REPO}/src/summarize_daily.py"   --date "${DATE}" --tickers "@${TICKERS_FILE}" --quiet

echo "=== DONE ${DATE} ==="
