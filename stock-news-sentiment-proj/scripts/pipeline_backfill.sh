#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="${1:-}"; END="${2:-}"

if [[ -z "${START}" || -z "${END}" ]]; then
  echo "usage: $0 START_DATE END_DATE   (YYYY-MM-DD YYYY-MM-DD)"
  exit 1
fi

# interpreter (python3 by default; respects PYTHON_BIN if exported)
PY="${PYTHON_BIN:-python3}"

# Build date list in Python (portable), read it without `mapfile`
DATES=()
while IFS= read -r line; do
  DATES+=("$line")
done < <("$PY" - "$START" "$END" <<'PY'
import sys, datetime as dt
s = dt.date.fromisoformat(sys.argv[1])
e = dt.date.fromisoformat(sys.argv[2])
d = s
while d <= e:
    print(d.isoformat())
    d += dt.timedelta(days=1)
PY
)

# Safety checks
if [[ ${#DATES[@]} -eq 0 ]]; then
  echo "No dates produced. Check your START/END."
  exit 1
fi

MAX_DAYS=366
if [[ ${#DATES[@]} -gt $MAX_DAYS ]]; then
  echo "Refusing to run for ${#DATES[@]} days (> $MAX_DAYS). Narrow your range."
  exit 1
fi

for D in "${DATES[@]}"; do
  "${REPO}/scripts/pipeline_daily.sh" "$D"
done

echo "=== BACKFILL DONE: ${START} → ${END} ==="

