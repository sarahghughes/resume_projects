# Factors & Option Pricing — Financial Economics Final

This repo contains a single-entry Python analysis that:
- Runs **CAPM** and **Fama–French 5-Factor (FF5)** regressions on `AAPL, JPM, XOM, WMT, XLV` over **2019–2024**.
- Generates clean **tables**, **plots**, and a short **insights** summary.
- Prices **PFE** options (European vs American) with a binomial tree under multiple volatility lookbacks.

> **Entry point:** `run_analysis.py`  
> **Outputs:** written under `outputs/` (overwritten on each run)  
> **FF5 factors CSV:** already included at `data/raw/ff5_daily_given.CSV`

---

## Repo layout

```
run_analysis.py
README.md
requirements.txt
.gitignore
data/
  raw/
    ff5_daily_given.CSV         # instructor-provided FF5 daily factors (included)
    .gitkeep
reports/
  financial_econ_report.pdf
  financial_econ_presentation.pdf
outputs/                        # generated each run (safe to delete)
  factors/
    data/
    tables/
    plots/
  options/
  insights/
```

---

## Quickstart

### 1) Environment
Tested on Python **3.10–3.12**.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
No setup needed. The FF5 daily CSV is **already in the repo** at:

```
data/raw/ff5_daily_given.CSV
```

- The loader auto-detects the first data row and parses flexibly.
- If the file uses decimals instead of percent, the script auto-scales to percent to match return units.

### 3) Run

```
python run_analysis.py
```

---

## What it produces

```
outputs/
  factors/
    data/
      aligned_returns_and_factors.csv
    tables/
      factor_regression_detailed.csv
      ff5_beta_pvalues.csv
    plots/
      factors_beta_comparison.png
      factors_r2_comparison.png
      factors_ff5_loadings.png
  options/
    pfe_binomial_prices.csv
    pfe_early_ex_summary.csv
  insights/
    INSIGHTS.md
```

**Notes**
- Files under `outputs/` are **overwritten** each run (clean for grading/portfolio).
- Inputs under `data/raw/` and your PDFs in `reports/` are **never modified**.
- Plots use **dynamic y-limits** so small betas remain readable.

---

## Method (brief)

**Factors**
1. Download adjusted closes via `yfinance`, compute daily percent returns.
2. Load FF5 daily factors (robust header detection), convert to percent if needed.
3. Join returns with factors; compute excess returns (`{ticker}_excess = return − RF`).
4. Run OLS:
   - **CAPM:** `excess ~ const + MKT_RF`
   - **FF5:**  `excess ~ const + MKT_RF + SMB + HML + RMW + CMA`
5. Save detailed table (betas, p-values, 95% CIs, R², ΔR²) and make plots.

**Options**
1. Pull **PFE** spot and realized vol from recent closes (annualize via √252).
2. Pull **^IRX** (13-week T-bill) as the risk-free proxy.
3. Price European & American calls/puts via CRR binomial tree for multiple lookbacks/strikes.
4. Save per-scenario prices and average early-exercise premia.

---

## Reproducibility

- `requirements.txt` pins major versions for stable results.
- No CLI flags—just run `python run_analysis.py`.
- Internet required for `yfinance` (prices + ^IRX). If offline, the run will fail on downloads.

---

## Troubleshooting

- **CSV parse error:** Use the original “given” FF5 daily CSV included in `data/raw/`. The loader finds the first `YYYYMMDD` line and skips headers automatically.
- **Network issues:** Retry; some networks block Yahoo Finance endpoints.

---

## License
MIT (or choose your preferred license).