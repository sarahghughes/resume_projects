#!/usr/bin/env python3
"""
run_analysis.py — Factors (CAPM/FF5) + Option Pricing, portfolio-clean outputs

Inputs
- data/raw/ff5_daily_given.CSV   # instructor-provided Fama-French 5 (daily)

Outputs
- outputs/factors/data/aligned_returns_and_factors.csv
- outputs/factors/tables/factor_regression_detailed.csv
- outputs/factors/tables/ff5_beta_pvalues.csv
- outputs/factors/plots/factors_beta_comparison.png
- outputs/factors/plots/factors_r2_comparison.png
- outputs/factors/plots/factors_ff5_loadings.png
- outputs/options/pfe_binomial_prices.csv
- outputs/options/pfe_early_ex_summary.csv
- outputs/insights/INSIGHTS.md
"""

from __future__ import annotations

import math
import os
import re
from datetime import date, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

# --------------------------
# Config (edit if needed)
# --------------------------
TICKERS: List[str] = ["AAPL", "JPM", "XOM", "WMT", "XLV"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
FF5_CSV_PATH = "data/raw/ff5_daily_given.CSV"  # match your actual filename

SHOW_PLOTS = False
OUTPUTS_ROOT = "outputs"
DIRS = {
    "factors_data": os.path.join(OUTPUTS_ROOT, "factors", "data"),
    "factors_tables": os.path.join(OUTPUTS_ROOT, "factors", "tables"),
    "factors_plots": os.path.join(OUTPUTS_ROOT, "factors", "plots"),
    "options": os.path.join(OUTPUTS_ROOT, "options"),
    "insights": os.path.join(OUTPUTS_ROOT, "insights"),
}

# --------------------------
# Utils
# --------------------------
def ensure_dirs() -> None:
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)

def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()  # headless & memory-light

# --------------------------
# FF5 robust loader
# --------------------------
def _detect_ff5_data_start(path: str) -> int:
    """Find first data line by YYYYMMDD token; handles long headers."""
    date_re = re.compile(r'^\s*"?(\d{8})"?')
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s or set(s) <= {",", ";"}:
                continue
            if date_re.match(s):
                return idx
    raise ValueError("Couldn't find a YYYYMMDD row in the FF5 CSV.")

def _load_ff5_csv_robust(ff5_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(ff5_csv_path):
        raise FileNotFoundError(
            f"Missing '{ff5_csv_path}'. Place the instructor-provided file under data/raw/."
        )
    skip = _detect_ff5_data_start(ff5_csv_path)
    df = pd.read_csv(
        ff5_csv_path,
        skiprows=skip,
        engine="python",
        sep=r"[,\s;]+",
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        on_bad_lines="skip",
    )
    df = df[pd.to_numeric(df["Date"], errors="coerce").notnull()].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(int), format="%Y%m%d")
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols).set_index("Date").sort_index()
    # If factors look like decimals, convert to percent (to match return units).
    if df[["MKT_RF", "RF"]].abs().median().min() < 0.02:
        df[cols] = df[cols] * 100.0
    return df

# --------------------------
# Prices + join
# --------------------------
def load_prices_and_ff5(
    tickers: List[str],
    start_date: str,
    end_date: str,
    ff5_csv_path: str = FF5_CSV_PATH,
) -> pd.DataFrame:
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No price data downloaded. Check tickers/dates/network.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            price_data = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            price_data = raw["Adj Close"]
        else:
            raise KeyError("Expected Close/Adj Close in columns.")
    else:
        price_data = raw.get("Adj Close", raw.get("Close"))
        if price_data is None:
            raise KeyError("Expected Close/Adj Close in columns.")

    returns = price_data.pct_change().dropna() * 100  # percent
    ff = _load_ff5_csv_robust(ff5_csv_path)
    data = returns.join(ff, how="inner")

    for t in tickers:
        data[f"{t}_excess"] = data[t] - data["RF"]

    ensure_dirs()
    data.to_csv(os.path.join(DIRS["factors_data"], "aligned_returns_and_factors.csv"))
    return data

# --------------------------
# Regressions
# --------------------------
def _ols_with_ci(y: pd.Series, X: pd.DataFrame) -> Dict[str, float]:
    model = sm.OLS(y, X).fit()
    out = {
        "alpha": float(model.params.get("const", float("nan"))),
        "alpha_p": float(model.pvalues.get("const", float("nan"))),
        "R2": float(model.rsquared),
        "R2_adj": float(model.rsquared_adj),
    }
    conf = model.conf_int(alpha=0.05)
    for name in model.params.index:
        out[name] = float(model.params[name])
        out[f"{name}_p"] = float(model.pvalues[name])
        if name in conf.index:
            out[f"{name}_ci_lo"] = float(conf.loc[name, 0])
            out[f"{name}_ci_hi"] = float(conf.loc[name, 1])
    return out

def run_factor_regressions_detailed(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    rows = []
    for t in tickers:
        y = data[f"{t}_excess"].dropna()
        idx = y.index
        capm = _ols_with_ci(y, sm.add_constant(data.loc[idx, ["MKT_RF"]]))
        ff5 = _ols_with_ci(y, sm.add_constant(data.loc[idx, factors]))
        rows.append({
            "Ticker": t,
            "CAPM_alpha": capm["alpha"],
            "CAPM_alpha_p": capm["alpha_p"],
            "CAPM_beta_MKT": capm.get("MKT_RF"),
            "CAPM_beta_MKT_p": capm.get("MKT_RF_p"),
            "CAPM_beta_MKT_ci_lo": capm.get("MKT_RF_ci_lo"),
            "CAPM_beta_MKT_ci_hi": capm.get("MKT_RF_ci_hi"),
            "CAPM_R2": capm["R2"],
            "CAPM_R2_adj": capm["R2_adj"],
            "FF5_alpha": ff5["alpha"],
            "FF5_alpha_p": ff5["alpha_p"],
            "FF5_beta_MKT": ff5.get("MKT_RF"),
            "FF5_beta_MKT_p": ff5.get("MKT_RF_p"),
            "FF5_beta_MKT_ci_lo": ff5.get("MKT_RF_ci_lo"),
            "FF5_beta_MKT_ci_hi": ff5.get("MKT_RF_ci_hi"),
            "FF5_beta_SMB": ff5.get("SMB"),
            "FF5_beta_SMB_p": ff5.get("SMB_p"),
            "FF5_beta_SMB_ci_lo": ff5.get("SMB_ci_lo"),
            "FF5_beta_SMB_ci_hi": ff5.get("SMB_ci_hi"),
            "FF5_beta_HML": ff5.get("HML"),
            "FF5_beta_HML_p": ff5.get("HML_p"),
            "FF5_beta_HML_ci_lo": ff5.get("HML_ci_lo"),
            "FF5_beta_HML_ci_hi": ff5.get("HML_ci_hi"),
            "FF5_beta_RMW": ff5.get("RMW"),
            "FF5_beta_RMW_p": ff5.get("RMW_p"),
            "FF5_beta_RMW_ci_lo": ff5.get("RMW_ci_lo"),
            "FF5_beta_RMW_ci_hi": ff5.get("RMW_ci_hi"),
            "FF5_beta_CMA": ff5.get("CMA"),
            "FF5_beta_CMA_p": ff5.get("CMA_p"),
            "FF5_beta_CMA_ci_lo": ff5.get("CMA_ci_lo"),
            "FF5_beta_CMA_ci_hi": ff5.get("CMA_ci_hi"),
            "FF5_R2": ff5["R2"],
            "FF5_R2_adj": ff5["R2_adj"],
            "Delta_R2": ff5["R2"] - capm["R2"],
        })
    return pd.DataFrame(rows)

def build_ff5_significance_table(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    rows = []
    for t in tickers:
        y = data[f"{t}_excess"].dropna()
        X = sm.add_constant(data.loc[y.index, factors])
        model = sm.OLS(y, X).fit()
        for f in factors:
            rows.append({"Ticker": t, "Factor": f, "Beta": model.params[f], "p_value": model.pvalues[f]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIRS["factors_tables"], "ff5_beta_pvalues.csv"), index=False)
    return df

# --------------------------
# Plots (dynamic y-limits)
# --------------------------
def _nice_ylim(values: List[float], pad_ratio: float = 0.2) -> Tuple[float, float]:
    vmin, vmax = float(min(values)), float(max(values))
    if vmin == vmax:
        vmin, vmax = vmin - 0.1, vmax + 0.1
    span = vmax - vmin
    return vmin - pad_ratio * span, vmax + pad_ratio * span

def make_factor_visuals(summary_df: pd.DataFrame, show: bool = SHOW_PLOTS) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    tickers = summary_df["Ticker"].tolist()
    x_pos = list(range(len(summary_df)))

    # Market betas — readable even when small
    capm_b = summary_df["CAPM_beta_MKT"].tolist()
    ff5_b = summary_df["FF5_beta_MKT"].tolist()
    plt.figure(figsize=(8, 5))
    plt.bar(x_pos, capm_b, alpha=0.6, label="CAPM β (MKT)")
    plt.plot(x_pos, ff5_b, marker="o", linestyle="--", label="FF5 β (MKT)")
    plt.xticks(x_pos, tickers)
    ylo, yhi = _nice_ylim(capm_b + ff5_b, pad_ratio=0.25)
    plt.ylim(ylo, yhi)
    if ylo <= 1.0 <= yhi:
        plt.axhline(1.0, linestyle=":", linewidth=1)
    plt.ylabel("Market Beta")
    plt.title("Market Beta by Ticker: CAPM vs Fama-French 5-Factor")
    plt.legend()
    p = os.path.join(DIRS["factors_plots"], "factors_beta_comparison.png")
    savefig(p); paths["beta_plot"] = p
    if show: plt.show()

    # R² comparison — interpretable [0,1]
    plt.figure(figsize=(8, 5))
    width = 0.35
    indices = x_pos
    plt.bar([i - width/2 for i in indices], summary_df["CAPM_R2"], width=width, label="CAPM R²")
    plt.bar([i + width/2 for i in indices], summary_df["FF5_R2"], width=width, label="FF5 R²")
    plt.xticks(indices, tickers)
    plt.ylim(0, 1)
    plt.ylabel("R-squared")
    plt.title("Model Fit: CAPM vs Fama-French 5-Factor")
    plt.legend()
    p = os.path.join(DIRS["factors_plots"], "factors_r2_comparison.png")
    savefig(p); paths["r2_plot"] = p
    if show: plt.show()

    # FF5 non-market loadings — tight span
    factors = ["FF5_beta_SMB", "FF5_beta_HML", "FF5_beta_RMW", "FF5_beta_CMA"]
    labels = ["SMB (Size)", "HML (Value)", "RMW (Profitability)", "CMA (Investment)"]
    all_vals: List[float] = []
    for c in factors: all_vals += summary_df[c].tolist()
    ylo, yhi = _nice_ylim(all_vals, pad_ratio=0.25)

    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    for j, (col, lab) in enumerate(zip(factors, labels)):
        positions = [i + (j - len(factors)/2)*bar_width + bar_width/2 for i in indices]
        plt.bar(positions, summary_df[col], width=bar_width, label=lab)
    plt.xticks(indices, tickers)
    plt.ylim(ylo, yhi)
    if ylo <= 0.0 <= yhi:
        plt.axhline(0.0, linewidth=1)
    plt.ylabel("Factor Loading")
    plt.title("Fama-French 5-Factor Exposures by Ticker")
    plt.legend()
    p = os.path.join(DIRS["factors_plots"], "factors_ff5_loadings.png")
    savefig(p); paths["loadings_plot"] = p
    if show: plt.show()

    return paths

# --------------------------
# Insights
# --------------------------
def write_insights_md(reg_df: pd.DataFrame, sig_df: pd.DataFrame) -> str:
    lines = ["# Insights", "", "## What the models say", ""]
    top_delta = reg_df.sort_values("Delta_R2", ascending=False).head(3)
    lines.append("**Top ΔR² (FF5 over CAPM):**")
    for _, r in top_delta.iterrows():
        lines.append(f"- {r['Ticker']}: ΔR² = {r['Delta_R2']:.3f} (CAPM {r['CAPM_R2']:.3f} → FF5 {r['FF5_R2']:.3f})")
    lines.append("")
    lines.append("**Significant non-market factors (p < 0.05):**")
    for t in reg_df["Ticker"]:
        sub = sig_df[
            (sig_df["Ticker"] == t)
            & (sig_df["Factor"].isin(["SMB", "HML", "RMW", "CMA"]))
            & (sig_df["p_value"] < 0.05)
        ]
        if not sub.empty:
            facs = ", ".join(f"{row.Factor} (β={row.Beta:.3f})" for _, row in sub.iterrows())
            lines.append(f"- {t}: {facs}")
    lines.append("")
    lines.append("**Alpha check (FF5):**")
    for _, r in reg_df.iterrows():
        sig = " (significant)" if r["FF5_alpha_p"] < 0.05 else ""
        lines.append(f"- {r['Ticker']}: α = {r['FF5_alpha']:.3f}% (p={r['FF5_alpha_p']:.3f}){sig}")

    path = os.path.join(DIRS["insights"], "INSIGHTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

# --------------------------
# Options (binomial)
# --------------------------
def get_pfe_market_inputs(
    valuation_date: date, vol_lookback_days: int = 60
) -> Tuple[float, float, float]:
    stock = yf.Ticker("PFE")
    start_date = valuation_date - timedelta(days=vol_lookback_days * 2)
    end_date = valuation_date + timedelta(days=1)
    hist = stock.history(start=start_date, end=end_date, interval="1d").dropna(subset=["Close"])
    if hist.empty:
        raise RuntimeError("No PFE data downloaded from Yahoo Finance.")
    S0 = float(hist["Close"].iloc[-1])
    returns = hist.tail(vol_lookback_days)["Close"].pct_change().dropna()
    sigma = (float(returns.std()) * math.sqrt(252)) if len(returns) else 0.5
    irx = yf.Ticker("^IRX")
    irx_hist = irx.history(
        start=valuation_date - timedelta(days=10),
        end=valuation_date + timedelta(days=1),
        interval="1d",
    ).dropna(subset=["Close"])
    r = float(irx_hist["Close"].iloc[-1]) / 100.0 if not irx_hist.empty else 0.04
    return S0, sigma, r

def binomial_european(S0: float, K: float, r: float, sigma: float, T: float, N: int, option_type: str="call", q: float=0.0) -> float:
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):  # ensure valid tree
        raise ValueError(f"Risk-neutral probability out of bounds: p = {p:.6f}")
    prices = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    option_type = option_type.lower()
    values = [max(price - K, 0.0) for price in prices] if option_type=="call" else [max(K - price, 0.0) for price in prices]
    for i in range(N - 1, -1, -1):
        values = [disc * (p * values[j + 1] + (1.0 - p) * values[j]) for j in range(i + 1)]
    return values[0]

def binomial_american(S0: float, K: float, r: float, sigma: float, T: float, N: int, option_type: str="call", q: float=0.0) -> float:
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral probability out of bounds: p = {p:.6f}")
    prices = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    option_type = option_type.lower()
    values = [max(price - K, 0.0) for price in prices] if option_type=="call" else [max(K - price, 0.0) for price in prices]
    for i in range(N - 1, -1, -1):
        prices = [S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        continuation = [disc * (p * values[j + 1] + (1.0 - p) * values[j]) for j in range(i + 1)]
        intrinsic = [max(price - K, 0.0) for price in prices] if option_type=="call" else [max(K - price, 0.0) for price in prices]
        values = [max(continuation[j], intrinsic[j]) for j in range(i + 1)]
    return values[0]

def run_pfe_binomial_experiment() -> pd.DataFrame:
    valuation_date = date(2025, 11, 26)
    expiry_date = date(2026, 11, 26)
    days_to_expiry = (expiry_date - valuation_date).days
    if days_to_expiry <= 0:
        raise RuntimeError("Expiry date must be in the future.")
    T = days_to_expiry / 365.0
    N = 500
    strikes = [25.0, 26.0, 27.0]
    q = 0.03
    vol_windows = [20, 40, 60]

    rows: List[Dict] = []
    for lookback in vol_windows:
        S0, sigma, r = get_pfe_market_inputs(valuation_date, vol_lookback_days=lookback)
        for K in strikes:
            euro_call = binomial_european(S0, K, r, sigma, T, N, "call", q=q)
            euro_put  = binomial_european(S0, K, r, sigma, T, N, "put",  q=q)
            amer_call = binomial_american(S0, K, r, sigma, T, N, "call", q=q)
            amer_put  = binomial_american(S0, K, r, sigma, T, N, "put",  q=q)
            rows += [
                {"lookback_days": lookback, "S0": S0, "sigma_ann": sigma, "r_ann": r, "q_div": q, "N": N,
                 "strike": K, "option_type": "call", "european_price": euro_call, "american_price": amer_call,
                 "early_ex_premium": amer_call - euro_call},
                {"lookback_days": lookback, "S0": S0, "sigma_ann": sigma, "r_ann": r, "q_div": q, "N": N,
                 "strike": K, "option_type": "put", "european_price": euro_put, "american_price": amer_put,
                 "early_ex_premium": amer_put - euro_put},
            ]
    df = pd.DataFrame(rows)
    os.makedirs(DIRS["options"], exist_ok=True)
    df.to_csv(os.path.join(DIRS["options"], "pfe_binomial_prices.csv"), index=False)
    summary = (
        df.groupby(["lookback_days", "option_type"])["early_ex_premium"]
        .mean().reset_index().rename(columns={"early_ex_premium": "avg_early_ex_premium"})
    )
    summary.to_csv(os.path.join(DIRS["options"], "pfe_early_ex_summary.csv"), index=False)
    return df

# --------------------------
# Main
# --------------------------
def main() -> None:
    ensure_dirs()

    # Factors
    print("=== PART I: FACTOR MODELS (CAPM + FF5) ===")
    data = load_prices_and_ff5(TICKERS, START_DATE, END_DATE)
    reg_df = run_factor_regressions_detailed(data, TICKERS)
    print(reg_df[["Ticker","CAPM_R2","FF5_R2","Delta_R2"]].round(4).to_string(index=False))

    reg_path = os.path.join(DIRS["factors_tables"], "factor_regression_detailed.csv")
    reg_df.to_csv(reg_path, index=False)
    sig_df = build_ff5_significance_table(data, TICKERS)

    plot_df = reg_df[[
        "Ticker","CAPM_beta_MKT","FF5_beta_MKT","CAPM_R2","FF5_R2",
        "FF5_beta_SMB","FF5_beta_HML","FF5_beta_RMW","FF5_beta_CMA"
    ]]
    plot_paths = make_factor_visuals(plot_df, show=SHOW_PLOTS)
    insights_path = write_insights_md(reg_df, sig_df)

    # Options
    print("\n=== PART II: PFE BINOMIAL OPTION PRICING ===")
    _ = run_pfe_binomial_experiment()

    print("\nSaved organized outputs under /outputs/")

if __name__ == "__main__":
    main()
