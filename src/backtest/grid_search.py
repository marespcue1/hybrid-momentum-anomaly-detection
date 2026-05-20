# =============================================================================
# src/backtest/grid_search.py
# =============================================================================

from __future__ import annotations

import itertools
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path bootstrap ────────────────────────────────────────────────────────────
_THIS_FILE    = os.path.abspath(__file__)
_BACKTEST_DIR = os.path.dirname(_THIS_FILE)    # src/backtest/
_SRC_DIR      = os.path.dirname(_BACKTEST_DIR)  # src/
_ROOT         = os.path.dirname(_SRC_DIR)        # project root

for _p in (_SRC_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backtest.engine import (   # type: ignore
    BacktestConfig,
    load_data,
    build_portfolio,
    run_benchmark,
    get_daily_benchmark,
    compute_metrics,
)
from config import DIR_RESULTS, ensure_dirs  # type: ignore


# =============================================================================
# CANONICAL PARAM GRID (spec-compliant)
# Passed as default when run_grid_search is called without a custom grid.
# =============================================================================

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "top_n":                  [3,4, 5,6,7],
    "rebalance_freq":         ["M", "Q"],
    "anomaly_lookback_days":  [1,3,5,10],
    "momentum_signal": ["ALR12M_SKIP_Z", "ALR6M_SKIP_Z", "ALR1M_Z", "ALR1W_Z"],
    "active_cluster":         [0,1,2, 3,4],
    "rank_weighted":          [False, True],
    "min_portfolio_size":     [2, 3,4,5],
}

# Alias de compatibilidad con walk_forward.py y run_pipeline.py
PARAM_GRID = DEFAULT_PARAM_GRID


# =============================================================================
# CORE GRID SEARCH FUNCTION
# =============================================================================

def run_grid_search(
    param_grid: dict[str, list[Any]] | None = None,
    *,
    date_range: tuple | None = None,
    plot: bool = False,
    save_csv: str | None = None,
    df_clusters_preloaded: pd.DataFrame | None = None,
    price_wide_preloaded:  pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exhaustive grid search over BacktestConfig parameters.

    Parameters
    ----------
    param_grid : dict
        Keys must match BacktestConfig field names.
        Defaults to DEFAULT_PARAM_GRID (spec-compliant).

    date_range : tuple | None
        Optional (start_str, end_str) applied to every backtest run.
        Pass FECHA_INICIO_ANALISIS / FECHA_TRAIN_END for In-Sample search.

    plot : bool
        When True, renders the Spaghetti Chart after the sweep.

    save_csv : str | None
        Optional path to export results CSV.

    df_clusters_preloaded / price_wide_preloaded
        Pre-loaded DataFrames — avoids re-reading CSVs when called from
        walk_forward.py (which passes the full dataset and handles slicing
        via date_range inside BacktestConfig).

    Returns
    -------
    pd.DataFrame — results sorted by Sharpe descending, one row per combo.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    print(f"\n{'='*70}")
    print("  GRID SEARCH — Hybrid Momentum Strategy")
    print(f"{'='*70}")
    if date_range:
        print(f"  Date range : {date_range[0]} → {date_range[1]}")

    # ── 1. Load data ONCE ────────────────────────────────────────────────────
    if df_clusters_preloaded is not None and price_wide_preloaded is not None:
        df_clusters = df_clusters_preloaded
        price_wide  = price_wide_preloaded
        print("  [1/4] Using pre-loaded DataFrames (no file I/O).")
    else:
        dummy_cfg = BacktestConfig(verbose=False, plot=False)
        print("  [1/4] Loading datasets into RAM (once)...")
        df_clusters, price_wide = load_data(dummy_cfg)

    # ── 2. Validate momentum_signal columns upfront ──────────────────────────
    signals_to_test: list[str] = param_grid.get("momentum_signal", ["ALR12M_SKIP_Z"])
    if not isinstance(signals_to_test, list):
        signals_to_test = [signals_to_test]

    available_cols = set(df_clusters.columns)
    for sig in signals_to_test:
        if sig not in available_cols:
            raise KeyError(
                f"[grid_search] momentum_signal='{sig}' not found in "
                f"05_dataset_clusters.csv.\n"
                f"  Available signal columns: "
                f"{sorted(c for c in available_cols if c.endswith('_Z'))}"
            )

    # ── 3. Generate combinations ─────────────────────────────────────────────
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total_combos = len(combos)
    print(f"  [2/4] Evaluating {total_combos:,} combinations × "
          f"{len(keys)} parameters...")
    print(f"        Parameters: {keys}\n")

    results:           list[dict]       = []
    equity_curves:     dict             = {}
    benchmarks_curves: dict             = {}
    spy_daily_global:  pd.Series | None = None

    # ── 4. Evaluation loop (no file I/O here) ────────────────────────────────
    start_time = time.time()
    print(f"  [3/4] Iniciando evaluación de {total_combos:,} combinaciones...")

    for idx, combo in enumerate(combos, 1):
        kwargs = dict(zip(keys, combo))

        # Build config — always disable verbose/plot inside loop
        cfg = BacktestConfig(
            verbose=False,
            plot=False,
            date_range=date_range,
            **kwargs,
        )

        # Apply date filter to pre-loaded DataFrames inline
        if date_range is not None:
            start_ts = pd.Timestamp(date_range[0])
            end_ts   = pd.Timestamp(date_range[1])
            df_cl_slice = df_clusters[
                (df_clusters["Fecha"] >= start_ts) &
                (df_clusters["Fecha"] <= end_ts)
            ]
            pw_slice = price_wide.loc[
                (price_wide.index >= start_ts) &
                (price_wide.index <= end_ts)
            ]
        else:
            df_cl_slice = df_clusters
            pw_slice    = price_wide

        # Pass pre-loaded DataFrames directly — zero CSV reads
        portfolio_df = build_portfolio(df_cl_slice, pw_slice, cfg)

        if portfolio_df.empty or portfolio_df["n_held"].sum() == 0:
            if idx % 200 == 0 or idx == total_combos:
                pct = (idx / total_combos) * 100
                print(f"    > Progreso: {idx}/{total_combos} ({pct:.1f}%)...")
            continue

        benchmark_df = run_benchmark(portfolio_df, pw_slice, cfg.benchmark_ticker)
        metrics      = compute_metrics(portfolio_df, benchmark_df, cfg.rebalance_freq)

        # Store global daily SPY once
        if spy_daily_global is None:
            spy_daily_global = get_daily_benchmark(
                portfolio_df, pw_slice, cfg.benchmark_ticker
            )

        # ── Build equity curve for this combo ────────────────────────────
        merged = (
            portfolio_df[["rebal_date", "portfolio_return"]]
            .merge(benchmark_df[["rebal_date", "benchmark_return"]],
                   on="rebal_date", how="inner")
            .dropna()
        )
        port_curve  = (1 + merged["portfolio_return"]).cumprod()
        bench_curve = (1 + merged["benchmark_return"]).cumprod()
        dates       = merged["rebal_date"]

        freq_key = cfg.rebalance_freq
        if freq_key not in benchmarks_curves:
            benchmarks_curves[freq_key] = (dates, bench_curve)

        combo_name = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        equity_curves[combo_name] = (dates, port_curve)

        # ── Accumulate metrics row ────────────────────────────────────────
        row = kwargs.copy()
        row.update({
            "CAGR (%)":      round((metrics.get("cagr_portfolio")   or np.nan) * 100, 2),
            "Sharpe":        round( metrics.get("sharpe_portfolio")  or np.nan,       3),
            "MaxDD (%)":     round((metrics.get("maxdd_portfolio")   or np.nan) * 100, 2),
            "HitRate (%)":   round((metrics.get("hit_rate_portfolio") or np.nan) * 100, 1),
            "ActivePeriods": metrics.get("n_periods_active", 0),
            "CashPeriods":   metrics.get("n_cash_periods",   0),
            "TotalPeriods":  metrics.get("n_periods_total",  0),
            "BenchCAGR (%)": round((metrics.get("cagr_benchmark")   or np.nan) * 100, 2),
            "BenchSharpe":   round( metrics.get("sharpe_benchmark")  or np.nan,       3),
        })
        results.append(row)

        if idx % 200 == 0 or idx == total_combos:
            pct = (idx / total_combos) * 100
            print(f"    > Progreso: {idx}/{total_combos} ({pct:.1f}%)...")

    elapsed = time.time() - start_time
    print(f"  [3/4] Sweep complete. Valid combinations: {len(results)}")
    print(f"  [!] Búsqueda finalizada en {elapsed:.2f} segundos.")

    if not results:
        print("  [WARNING] No valid results. Check param_grid and date_range.")
        return pd.DataFrame()

    # ── 5. Build results table ────────────────────────────────────────────────
    results_df = (
        pd.DataFrame(results)
        .sort_values("Sharpe", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n{'─'*70}")
    print("  TOP 10 COMBINATIONS (by Sharpe Ratio)")
    print(f"{'─'*70}")
    display_cols = keys + ["CAGR (%)", "Sharpe", "MaxDD (%)", "HitRate (%)", "ActivePeriods"]
    print(results_df[display_cols].head(10).to_string(index=True))
    print(f"{'─'*70}")

    if save_csv:
        ensure_dirs()
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        results_df.to_csv(save_csv, sep=";", decimal=",", index=False)
        print(f"\n  Results saved → {save_csv}")

    # ── 6. Spaghetti Chart ───────────────────────────────────────────────────
    if plot and equity_curves:
        print(f"  [4/4] Rendering Spaghetti Chart...")
        _plot_spaghetti(
            results_df=results_df,
            equity_curves=equity_curves,
            benchmarks_curves=benchmarks_curves,
            spy_daily_global=spy_daily_global,
            keys=keys,
        )

    return results_df


# =============================================================================
# SPAGHETTI CHART
# =============================================================================

def _plot_spaghetti(
    results_df: pd.DataFrame,
    equity_curves: dict,
    benchmarks_curves: dict,
    spy_daily_global: pd.Series | None,
    keys: list[str],
) -> None:
    """Render top-50 equity curves, color-coded by active_cluster."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped — matplotlib not available]")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    cluster_colors = {
        0: "#d62728", 1: "#9467bd", 2: "#2ca02c", 3: "#ff7f0e", 4: "#7f7f7f"
    }

    top_names: list[str] = []
    for i in range(min(50, len(results_df))):
        row  = results_df.iloc[i]
        name = " | ".join(f"{k}={row[k]}" for k in keys)
        top_names.append(name)

    if not top_names:
        return

    best_name       = top_names[0]
    plotted_clusters: set = set()

    for name in top_names:
        if name not in equity_curves:
            continue
        dates, curve = equity_curves[name]

        cluster_val = -1
        for part in name.split(" | "):
            if "active_cluster=" in part:
                try:
                    cluster_val = int(part.split("=")[1])
                except ValueError:
                    pass

        color     = cluster_colors.get(cluster_val, "gray")
        alpha     = 1.0 if name == best_name else 0.3
        lw        = 3.5 if name == best_name else 0.8
        zorder    = 10  if name == best_name else 5
        label     = f"Cluster {cluster_val}" if cluster_val not in plotted_clusters else None
        if label:
            plotted_clusters.add(cluster_val)

        ax.plot(dates, curve, color=color, alpha=alpha, linewidth=lw,
                label=label, zorder=zorder)

    if spy_daily_global is not None and not spy_daily_global.empty:
        ax.plot(spy_daily_global.index, spy_daily_global,
                color="black", linewidth=1.5, label="SPY (daily)", alpha=0.6)

    ax.set_title(f"Grid Search: Top 50 de {len(results_df)} combinaciones", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Cumulative Return (base=1)")

    ensure_dirs()
    out = os.path.join(DIR_RESULTS, "grid_search_spaghetti.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] Spaghetti chart saved → {out}")
    plt.close()


# =============================================================================
# ENTRY POINT (direct execution — In-Sample only)
# =============================================================================

if __name__ == "__main__":
    from config import FECHA_INICIO_ANALISIS, FECHA_TRAIN_END  # type: ignore

    ensure_dirs()
    out_csv = os.path.join(DIR_RESULTS, "grid_search_insample.csv")

    df_results = run_grid_search(
        param_grid=DEFAULT_PARAM_GRID,
        date_range=(FECHA_INICIO_ANALISIS, FECHA_TRAIN_END),
        plot=True,
        save_csv=out_csv,
    )
    print(f"\nGrid search complete. {len(df_results)} valid combinations.")
