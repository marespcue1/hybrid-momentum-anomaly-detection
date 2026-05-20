# =============================================================================
# src/backtest/engine.py
# =============================================================================

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path bootstrap — works whether called from root or from src/
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
_BACKTEST_DIR = os.path.dirname(_THIS_FILE)   # src/backtest/
_SRC_DIR      = os.path.dirname(_BACKTEST_DIR) # src/
_ROOT         = os.path.dirname(_SRC_DIR)       # project root

for _p in (_SRC_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # type: ignore
    FILE_CLUSTERS,
    FILE_PRICES,
    DIR_RESULTS,
    DIR_FIGURES,
    ensure_dirs,
)

BENCHMARK_TICKER = "SPY"


# =============================================================================
# 1. CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class BacktestConfig:
    """All tunable parameters for the backtest.
    Pass explicit kwargs or accept defaults; never hard-code inside functions.
    """

    # ── Core strategy ────────────────────────────────────────────────────────
    top_n:           int                        = 5
    rebalance_freq:  Literal["M", "Q"]          = "M"
    active_cluster:  int                        = 2
    momentum_signal: str                        = "ALR12M_SKIP_Z"

    # ── DBSCAN anomaly guard ─────────────────────────────────────────────────
    # Asset excluded at T if DBSCAN_Label == -1 in preceding N calendar days
    anomaly_lookback_days: int                  = 20

    # ── Portfolio minimum size ───────────────────────────────────────────────
    # If eligible assets after cluster + anomaly filter < this → go to CASH
    min_portfolio_size: int                     = 1

    # ── Portfolio weighting ─────────────────────────────────────────────────
    # False → equal weight (1/N)
    # True  → rank-decay: weight ∝ (N+1-rank) / sum(1..N)
    rank_weighted: bool                         = False

    # ── Benchmark ────────────────────────────────────────────────────────────
    benchmark_ticker: str                       = BENCHMARK_TICKER

    # ── Walk-Forward date filter ─────────────────────────────────────────────
    # Tuple of (start_date_str, end_date_str) — None means use all data
    date_range: tuple | None                    = None

    # ── Paths ────────────────────────────────────────────────────────────────
    path_clusters: str = field(default_factory=lambda: FILE_CLUSTERS)
    path_bruto:    str = field(default_factory=lambda: FILE_PRICES)

    # ── Output ───────────────────────────────────────────────────────────────
    verbose: bool = True
    plot:    bool = True


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def _read_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[engine] File not found: {path}\n  ({label})"
        )
    return pd.read_csv(path, sep=";", decimal=",", low_memory=False)


def load_data(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two source files once and apply optional date_range filter.

    Returns
    -------
    df_clusters : Fecha, Ticker, Cluster, DBSCAN_Label, *_Z features
    price_wide  : wide price table (index=Fecha, columns=Tickers)

    Note
    ────
    05_dataset_clusters.csv is the single source for both regime (Cluster)
    and anomaly (DBSCAN_Label) information.  No separate DBSCAN file needed.
    """
    df_clusters = _read_csv(cfg.path_clusters, "K-Means + DBSCAN labels")
    df_bruto    = _read_csv(cfg.path_bruto,    "Raw prices")

    df_clusters["Fecha"] = pd.to_datetime(df_clusters["Fecha"])
    df_bruto["Fecha"]    = pd.to_datetime(df_bruto["Fecha"])

    # ── Apply date_range filter BEFORE anything else ──────────────────────
    if cfg.date_range is not None:
        start, end = pd.Timestamp(cfg.date_range[0]), pd.Timestamp(cfg.date_range[1])
        df_clusters = df_clusters[
            (df_clusters["Fecha"] >= start) & (df_clusters["Fecha"] <= end)
        ].copy()
        df_bruto = df_bruto[
            (df_bruto["Fecha"] >= start) & (df_bruto["Fecha"] <= end)
        ].copy()

    # Verify DBSCAN_Label is present in the clusters file
    if "DBSCAN_Label" not in df_clusters.columns:
        raise KeyError(
            "[engine] 'DBSCAN_Label' column not found in 05_dataset_clusters.csv. "
            "Ensure the preprocessing pipeline has merged DBSCAN labels into this file."
        )

    price_wide = (
        df_bruto[["Fecha", "Ticker", "Precio_Adj"]]
        .drop_duplicates(["Fecha", "Ticker"])
        .pivot(index="Fecha", columns="Ticker", values="Precio_Adj")
        .sort_index()
    )
    price_wide.columns.name = None

    if cfg.verbose:
        date_tag = (
            f"{cfg.date_range[0]} → {cfg.date_range[1]}"
            if cfg.date_range else "full history"
        )
        print(f"\n{'='*62}")
        print("  DATA LOADED")
        print(f"{'='*62}")
        print(f"  Window   : {date_tag}")
        print(f"  Clusters : {len(df_clusters):>8,} rows | "
              f"{df_clusters['Fecha'].min().date()} → {df_clusters['Fecha'].max().date()}")
        print(f"  Prices   : {price_wide.shape[0]:>8,} dates × "
              f"{price_wide.shape[1]} tickers")
        print(f"{'='*62}")

    return df_clusters, price_wide


# =============================================================================
# 3. REBALANCE DATE GENERATION
# =============================================================================

def get_rebalance_dates(
    price_wide: pd.DataFrame,
    freq: Literal["M", "Q"],
) -> pd.DatetimeIndex:
    """Last actual trading day in each month (M) or quarter (Q).

    Zero look-ahead: derived entirely from the observed price index.
    No synthetic calendar dates are ever generated.
    """
    trading_dates = pd.DatetimeIndex(price_wide.index)
    period_labels = trading_dates.to_period("M" if freq == "M" else "Q")

    rebal_series = (
        pd.Series(trading_dates, index=period_labels)
        .groupby(level=0)
        .last()
    )
    return pd.DatetimeIndex(rebal_series.values)


# =============================================================================
# 4. HELPER: ENTRY / EXIT DATE RESOLUTION
# =============================================================================

def _get_entry_exit_dates(
    rebal_date: pd.Timestamp,
    next_rebal_date: pd.Timestamp,
    price_wide: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Entry = first trading day AFTER T.  Exit = last trading day ≤ T_next.

    Returns (None, None) when no valid window exists.
    """
    all_dates = price_wide.index

    entry_candidates = all_dates[all_dates > rebal_date]
    if len(entry_candidates) == 0:
        return None, None
    entry_date = entry_candidates[0]

    exit_candidates = all_dates[all_dates <= next_rebal_date]
    if len(exit_candidates) == 0:
        return None, None
    exit_date = exit_candidates[-1]

    if exit_date <= entry_date:
        return None, None

    return entry_date, exit_date


# =============================================================================
# 5. WEIGHT COMPUTATION
# =============================================================================

def _compute_weights(n: int, rank_weighted: bool) -> np.ndarray:
    """Return a weight vector of length n.

    Equal weight  : [1/n, 1/n, ...]
    Rank-decaying : rank-1 asset gets the most weight.
                    w_i ∝ (n + 1 - i)  for i = 1..n  (rank-1 indexed)
                    Normalized so weights sum to 1.
    """
    if not rank_weighted or n == 1:
        return np.full(n, 1.0 / n)
    ranks = np.arange(1, n + 1, dtype=float)
    raw   = (n + 1) - ranks
    return raw / raw.sum()


# =============================================================================
# 6. ASSET SELECTION (one rebalance date)
# =============================================================================

def _select_assets(
    rebal_date: pd.Timestamp,
    df_clusters: pd.DataFrame,
    cfg: BacktestConfig,
) -> list[str]:
    """Return ordered list of tickers using only information at T = rebal_date.

    Steps (all strictly causal):
      1. Regime filter  : Cluster == cfg.active_cluster
      2. Anomaly filter : no DBSCAN_Label == -1 in last N calendar days
      3. min_portfolio_size check → returns [] (→ CASH) if too few remain
      4. Momentum rank  : descending by cfg.momentum_signal
      5. Top-N selection
    """
    # (1) Regime filter
    cs = df_clusters[df_clusters["Fecha"] == rebal_date]
    if cs.empty:
        return []

    regime_ok = cs[cs["Cluster"] == cfg.active_cluster].copy()
    if regime_ok.empty:
        return []

    # (2) Anomaly filter — uses DBSCAN_Label from the same clusters file
    window_start = rebal_date - pd.Timedelta(days=cfg.anomaly_lookback_days)
    anomalous = df_clusters[
        (df_clusters["Fecha"] >= window_start) &
        (df_clusters["Fecha"] <= rebal_date) &
        (df_clusters["DBSCAN_Label"] == -1)
    ]["Ticker"].unique()

    regime_ok = regime_ok[~regime_ok["Ticker"].isin(anomalous)]

    # (3) min_portfolio_size guard → CASH
    if len(regime_ok) < cfg.min_portfolio_size:
        return []

    # (4) Momentum rank
    sig = cfg.momentum_signal
    if sig not in regime_ok.columns:
        raise KeyError(
            f"[engine] Signal '{sig}' not in 05_dataset_clusters.csv. "
            f"Available _Z columns: "
            f"{sorted(c for c in regime_ok.columns if c.endswith('_Z'))}"
        )
    regime_ok = regime_ok.dropna(subset=[sig]).sort_values(sig, ascending=False)

    # (5) Top-N
    return regime_ok.head(cfg.top_n)["Ticker"].tolist()


# =============================================================================
# 7. PORTFOLIO CONSTRUCTION (vectorized loop)
# =============================================================================

def build_portfolio(
    df_clusters: pd.DataFrame,
    price_wide: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Vectorized rebalancing loop.

    Returns one row per rebalance period with columns:
        rebal_date, entry_date, exit_date, tickers_held,
        n_held, portfolio_return, log_portfolio_return, weights_used

    A period where the strategy goes to CASH (no eligible assets or
    min_portfolio_size not met) records n_held=0, portfolio_return=0.0.

    The returned DataFrame is trimmed to start from the first period in which
    the strategy actually holds at least one asset (warm-up purge).
    """
    rebal_dates = get_rebalance_dates(price_wide, cfg.rebalance_freq)
    records: list[dict] = []

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal_date = rebal_dates[i + 1]

        entry_date, exit_date = _get_entry_exit_dates(
            rebal_date, next_rebal_date, price_wide
        )
        if entry_date is None:
            continue

        selected_tickers = _select_assets(rebal_date, df_clusters, cfg)

        # CASH period
        if not selected_tickers:
            records.append({
                "rebal_date":           rebal_date,
                "entry_date":           entry_date,
                "exit_date":            exit_date,
                "tickers_held":         [],
                "n_held":               0,
                "portfolio_return":     0.0,
                "log_portfolio_return": 0.0,
                "weights_used":         [],
                "cash_period":          True,
            })
            continue

        # Filter to tickers present in price table
        available = [t for t in selected_tickers if t in price_wide.columns]
        if not available:
            records.append({
                "rebal_date":           rebal_date,
                "entry_date":           entry_date,
                "exit_date":            exit_date,
                "tickers_held":         [],
                "n_held":               0,
                "portfolio_return":     0.0,
                "log_portfolio_return": 0.0,
                "weights_used":         [],
                "cash_period":          True,
            })
            continue

        prices_entry = price_wide.loc[entry_date, available]
        prices_exit  = price_wide.loc[exit_date,  available]

        valid_mask    = prices_entry.notna() & prices_exit.notna()
        valid_tickers = [t for t in available if valid_mask[t]]

        if not valid_tickers:
            records.append({
                "rebal_date":           rebal_date,
                "entry_date":           entry_date,
                "exit_date":            exit_date,
                "tickers_held":         [],
                "n_held":               0,
                "portfolio_return":     0.0,
                "log_portfolio_return": 0.0,
                "weights_used":         [],
                "cash_period":          True,
            })
            continue

        n = len(valid_tickers)
        weights = _compute_weights(n, cfg.rank_weighted)

        pe = prices_entry[valid_tickers].values.astype(float)
        px = prices_exit[valid_tickers].values.astype(float)
        log_returns = np.log(px / pe)

        log_port_ret    = float(np.dot(weights, log_returns))
        simple_port_ret = float(np.expm1(log_port_ret))

        records.append({
            "rebal_date":           rebal_date,
            "entry_date":           entry_date,
            "exit_date":            exit_date,
            "tickers_held":         valid_tickers,
            "n_held":               n,
            "portfolio_return":     simple_port_ret,
            "log_portfolio_return": log_port_ret,
            "weights_used":         weights.tolist(),
            "cash_period":          False,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # ── Warm-up trim: drop leading idle periods ───────────────────────────
    first_active_idx = df[df["n_held"] > 0].index.min()
    if pd.isna(first_active_idx):
        return df  # strategy never traded

    df = df.loc[first_active_idx:].reset_index(drop=True)
    return df


# =============================================================================
# 8. BENCHMARK
# =============================================================================

def run_benchmark(
    portfolio_df: pd.DataFrame,
    price_wide: pd.DataFrame,
    ticker: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Period benchmark returns aligned to portfolio_df rebalance windows."""
    if ticker not in price_wide.columns:
        raise ValueError(
            f"[engine] Benchmark '{ticker}' not in price data. "
            f"Available: {list(price_wide.columns)}"
        )

    rows = []
    for _, row in portfolio_df.iterrows():
        entry, exit_ = row["entry_date"], row["exit_date"]
        p_in  = price_wide.at[entry, ticker] if entry  in price_wide.index else np.nan
        p_out = price_wide.at[exit_, ticker] if exit_  in price_wide.index else np.nan

        if pd.notna(p_in) and pd.notna(p_out) and p_in > 0:
            log_r    = float(np.log(p_out / p_in))
            simple_r = float(np.expm1(log_r))
        else:
            log_r = simple_r = np.nan

        rows.append({
            "rebal_date":           row["rebal_date"],
            "benchmark_return":     simple_r,
            "log_benchmark_return": log_r,
        })
    return pd.DataFrame(rows)


def get_daily_benchmark(
    portfolio_df: pd.DataFrame,
    price_wide: pd.DataFrame,
    ticker: str = BENCHMARK_TICKER,
) -> pd.Series:
    """Daily price series normalized to 1.0, aligned to backtest start/end."""
    if ticker not in price_wide.columns or portfolio_df.empty:
        return pd.Series(dtype=float)

    start_date = portfolio_df["entry_date"].iloc[0]
    end_date   = portfolio_df["exit_date"].iloc[-1]

    spy_daily = price_wide.loc[start_date:end_date, ticker].dropna()
    if spy_daily.empty:
        return pd.Series(dtype=float)

    return spy_daily / spy_daily.iloc[0]


# =============================================================================
# 9. PERFORMANCE METRICS
# =============================================================================

def _sharpe(log_rets: pd.Series, ppy: int) -> float:
    clean = log_rets.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return np.nan
    return float((clean.mean() / clean.std(ddof=1)) * np.sqrt(ppy))


def _max_drawdown(simple_rets: pd.Series) -> float:
    cum = (1 + simple_rets.fillna(0)).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def _cagr(simple_rets: pd.Series, ppy: int) -> float:
    n = len(simple_rets.dropna())
    if n == 0:
        return np.nan
    total = (1 + simple_rets.fillna(0)).prod()
    return float(total ** (ppy / n) - 1)


def _hit_rate(simple_rets: pd.Series) -> float:
    valid = simple_rets.dropna()
    return float((valid > 0).mean()) if len(valid) > 0 else np.nan


def compute_metrics(
    portfolio_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    rebalance_freq: Literal["M", "Q"],
) -> dict:
    """Annualized Sharpe, MaxDD, CAGR, HitRate for portfolio and benchmark."""
    ppy = 12 if rebalance_freq == "M" else 4

    p_log  = portfolio_df["log_portfolio_return"]
    p_simp = portfolio_df["portfolio_return"]
    b_log  = benchmark_df["log_benchmark_return"]
    b_simp = benchmark_df["benchmark_return"]

    return {
        "sharpe_portfolio":   _sharpe(p_log,  ppy),
        "sharpe_benchmark":   _sharpe(b_log,  ppy),
        "maxdd_portfolio":    _max_drawdown(p_simp),
        "maxdd_benchmark":    _max_drawdown(b_simp),
        "cagr_portfolio":     _cagr(p_simp, ppy),
        "cagr_benchmark":     _cagr(b_simp, ppy),
        "hit_rate_portfolio": _hit_rate(p_simp),
        "n_periods_active":   int(portfolio_df["n_held"].gt(0).sum()),
        "n_periods_total":    len(portfolio_df),
        "n_cash_periods":     int(portfolio_df.get("cash_period", pd.Series(False)).sum()),
    }


# =============================================================================
# 10. CONSOLE REPORT
# =============================================================================

def _fp(val: float, d: int = 2) -> str:
    return f"{val*100:.{d}f}%" if pd.notna(val) else "  N/A"

def _ff(val: float, d: int = 3) -> str:
    return f"{val:.{d}f}" if pd.notna(val) else "  N/A"


def report(
    portfolio_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    benchmark_daily: pd.Series,
    metrics: dict,
    cfg: BacktestConfig,
) -> None:
    """Console performance table + optional equity-curve chart."""
    freq_label   = "Monthly" if cfg.rebalance_freq == "M" else "Quarterly"
    weight_label = "Rank-Decay" if cfg.rank_weighted else "Equal"
    date_tag     = (
        f"{cfg.date_range[0]} → {cfg.date_range[1]}"
        if cfg.date_range else "full history"
    )

    print(f"\n{'='*62}")
    print(f"  BACKTEST RESULTS — Hybrid Momentum (Walk-Forward)")
    print(f"{'='*62}")
    print(f"  Period       : {date_tag}")
    print(f"  Rebalancing  : {freq_label}")
    print(f"  Regime       : Cluster {cfg.active_cluster}")
    print(f"  Top-N        : {cfg.top_n}  ({weight_label} weighted)")
    print(f"  Signal       : {cfg.momentum_signal}")
    print(f"  Anomaly guard: {cfg.anomaly_lookback_days} cal. days (DBSCAN)")
    print(f"  Min portfolio: {cfg.min_portfolio_size} assets (else → CASH)")
    print(f"  Benchmark    : {cfg.benchmark_ticker}")
    print(f"{'─'*62}")
    print(f"  {'Metric':<32} {'Portfolio':>12} {'Benchmark':>12}")
    print(f"{'─'*62}")
    print(f"  {'CAGR':<32} {_fp(metrics['cagr_portfolio']):>12} {_fp(metrics['cagr_benchmark']):>12}")
    print(f"  {'Sharpe (annualized)':<32} {_ff(metrics['sharpe_portfolio']):>12} {_ff(metrics['sharpe_benchmark']):>12}")
    print(f"  {'Max Drawdown':<32} {_fp(metrics['maxdd_portfolio']):>12} {_fp(metrics['maxdd_benchmark']):>12}")
    print(f"  {'Hit Rate':<32} {_fp(metrics['hit_rate_portfolio']):>12}             —")
    print(f"  {'Active / Total / Cash periods':<32} "
          f"{metrics['n_periods_active']:>4,}/{metrics['n_periods_total']:<4,}/"
          f"{metrics.get('n_cash_periods', 0):<4,}")
    print(f"{'='*62}")

    if cfg.plot:
        try:
            import matplotlib.pyplot as plt
            _plot_equity_curves(portfolio_df, benchmark_df, benchmark_daily, cfg, plt)
        except ImportError:
            print("  [plot skipped — matplotlib not available]")


def _plot_equity_curves(
    portfolio_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    benchmark_daily: pd.Series,
    cfg: BacktestConfig,
    plt,
) -> None:
    merged = (
        portfolio_df[["rebal_date", "portfolio_return"]]
        .merge(benchmark_df[["rebal_date", "benchmark_return"]], on="rebal_date", how="inner")
        .dropna()
    )

    port_curve  = (1 + merged["portfolio_return"]).cumprod()
    bench_curve = (1 + merged["benchmark_return"]).cumprod()
    dates       = merged["rebal_date"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(dates, port_curve, label="Hybrid Momentum",
                 linewidth=2.5, color="#1f4e79", zorder=5)
    axes[0].plot(dates, bench_curve,
                 label=f"{cfg.benchmark_ticker} (period rebal.)",
                 linewidth=1.5, linestyle="--", color="#cc0000", zorder=4)

    if not benchmark_daily.empty:
        axes[0].plot(benchmark_daily.index, benchmark_daily,
                     label=f"{cfg.benchmark_ticker} (daily)",
                     linewidth=1.0, color="black", alpha=0.55, zorder=3)

    weight_tag = "Rank-Decay" if cfg.rank_weighted else "EW"
    title = (
        f"Equity Curve — {cfg.rebalance_freq} | "
        f"Top-{cfg.top_n} | {weight_tag} | Cluster {cfg.active_cluster}"
    )
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative return (base = 1)")

    port_max = port_curve.cummax()
    dd = (port_curve - port_max) / port_max
    axes[1].fill_between(dates, dd, 0, alpha=0.40, color="red", label="Portfolio DD")
    axes[1].set_title("Portfolio Drawdown")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel("Drawdown")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 11. MAIN ENTRYPOINT — run()
# =============================================================================

def run(
    cfg: BacktestConfig | None = None,
    *,
    df_clusters: pd.DataFrame | None = None,
    price_wide: pd.DataFrame | None = None,
) -> dict:
    """Execute the full backtest pipeline.

    Parameters
    ----------
    cfg        : BacktestConfig (created with defaults if None)
    df_clusters: pre-loaded clusters DataFrame (skips file I/O if provided)
    price_wide : pre-loaded wide price DataFrame (skips file I/O if provided)

    Returns
    -------
    dict with keys:
        portfolio_df     — one row per rebalance period
        benchmark_df     — periodic benchmark returns
        benchmark_daily  — daily benchmark series normalized to 1.0
        metrics          — performance dict
    """
    if cfg is None:
        cfg = BacktestConfig()

    if cfg.verbose:
        print(f"\n{'='*62}")
        print("  HYBRID MOMENTUM BACKTEST — STARTING")
        print(f"{'='*62}")
        print(f"  top_n={cfg.top_n} | freq={cfg.rebalance_freq} | "
              f"cluster={cfg.active_cluster} | signal={cfg.momentum_signal}")
        print(f"  rank_weighted={cfg.rank_weighted} | "
              f"min_portfolio_size={cfg.min_portfolio_size} | "
              f"anomaly_lookback={cfg.anomaly_lookback_days}d")

    # Load data if not pre-loaded (grid-search passes pre-loaded DFs)
    if df_clusters is None or price_wide is None:
        df_clusters, price_wide = load_data(cfg)
    elif cfg.date_range is not None:
        # Apply date filter to pre-loaded DataFrames
        start, end = pd.Timestamp(cfg.date_range[0]), pd.Timestamp(cfg.date_range[1])
        df_clusters = df_clusters[
            (df_clusters["Fecha"] >= start) & (df_clusters["Fecha"] <= end)
        ].copy()
        price_wide = price_wide.loc[
            (price_wide.index >= start) & (price_wide.index <= end)
        ].copy()

    rebal_dates = get_rebalance_dates(price_wide, cfg.rebalance_freq)
    if cfg.verbose:
        print(f"  Rebalance dates: {len(rebal_dates):,} "
              f"({rebal_dates[0].date()} → {rebal_dates[-1].date()})")

    portfolio_df = build_portfolio(df_clusters, price_wide, cfg)

    if portfolio_df.empty:
        if cfg.verbose:
            print("  [WARNING] Strategy produced zero trades. "
                  "Check cluster/signal availability for the given date_range.")
        return {
            "portfolio_df":    portfolio_df,
            "benchmark_df":    pd.DataFrame(),
            "benchmark_daily": pd.Series(dtype=float),
            "metrics":         {},
        }

    if cfg.verbose:
        active = portfolio_df["n_held"].gt(0).sum()
        print(f"  Portfolio: {len(portfolio_df):,} periods | {active:,} active")
        print(f"  First trade: {portfolio_df['entry_date'].iloc[0].date()} → "
              f"{portfolio_df['exit_date'].iloc[0].date()}")

    benchmark_df    = run_benchmark(portfolio_df, price_wide, cfg.benchmark_ticker)
    benchmark_daily = get_daily_benchmark(portfolio_df, price_wide, cfg.benchmark_ticker)
    metrics         = compute_metrics(portfolio_df, benchmark_df, cfg.rebalance_freq)

    if cfg.verbose:
        report(portfolio_df, benchmark_df, benchmark_daily, metrics, cfg)

    return {
        "portfolio_df":    portfolio_df,
        "benchmark_df":    benchmark_df,
        "benchmark_daily": benchmark_daily,
        "metrics":         metrics,
    }


# =============================================================================
# ENTRY POINT (direct execution)
# =============================================================================

if __name__ == "__main__":
    from config import FECHA_INICIO_ANALISIS, FECHA_TRAIN_END  # type: ignore

    cfg_champion = BacktestConfig(
        top_n=3,
        rebalance_freq="Q",
        active_cluster=3,
        momentum_signal="ALR12M_SKIP_Z",
        anomaly_lookback_days=20,
        rank_weighted=True,
        min_portfolio_size=1,
        date_range=(FECHA_INICIO_ANALISIS, FECHA_TRAIN_END),
        verbose=True,
        plot=True,
    )
    run(cfg_champion)
