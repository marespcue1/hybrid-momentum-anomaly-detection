# =============================================================================
# src/backtest/walk_forward.py
# =============================================================================

from __future__ import annotations

import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path bootstrap ────────────────────────────────────────────────────────────
_THIS_FILE    = os.path.abspath(__file__)
_BACKTEST_DIR = os.path.dirname(_THIS_FILE)
_SRC_DIR      = os.path.dirname(_BACKTEST_DIR)
_ROOT         = os.path.dirname(_SRC_DIR)

for _p in (_SRC_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # type: ignore
    FECHA_INICIO_ANALISIS,
    FECHA_TRAIN_END,
    FECHA_TEST_START,
    PORCENTAJE_TOP_GANADORES,
    DIR_RESULTS,
    FILE_CLUSTERS,
    FILE_PRICES,
    ensure_dirs,
)
from backtest.engine import (  # type: ignore
    BacktestConfig,
    load_data,
    build_portfolio,
    run_benchmark,
    compute_metrics,
    get_daily_benchmark,
)
from backtest.grid_search import run_grid_search, DEFAULT_PARAM_GRID  # type: ignore


# =============================================================================
# HELPERS
# =============================================================================

def _load_full_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full (unfiltered) datasets once for reuse across all splits."""
    dummy_cfg = BacktestConfig(verbose=False, plot=False)
    print("  Loading full datasets into RAM (once)...")
    df_clusters, price_wide = load_data(dummy_cfg)
    return df_clusters, price_wide


def _run_single(
    params: dict,
    date_range: tuple,
    df_clusters: pd.DataFrame,
    price_wide: pd.DataFrame,
) -> dict:
    """Run one backtest config on a given date_range using pre-loaded data.

    Returns the metrics dict (empty if no trades).
    """
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

    cfg = BacktestConfig(
        verbose=False,
        plot=False,
        date_range=date_range,
        **params,
    )

    portfolio_df = build_portfolio(df_cl_slice, pw_slice, cfg)
    if portfolio_df.empty or portfolio_df["n_held"].sum() == 0:
        return {}

    benchmark_df = run_benchmark(portfolio_df, pw_slice, cfg.benchmark_ticker)
    metrics      = compute_metrics(portfolio_df, benchmark_df, cfg.rebalance_freq)
    return metrics


def _save_equity_curve(
    params: dict,
    date_range: tuple,
    df_clusters: pd.DataFrame,
    price_wide: pd.DataFrame,
    out_dir: str,
    label: str,
) -> None:
    """Save a per-period equity curve CSV for a single config + date_range."""
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

    cfg = BacktestConfig(verbose=False, plot=False, date_range=date_range, **params)
    portfolio_df = build_portfolio(df_cl_slice, pw_slice, cfg)
    if portfolio_df.empty:
        return

    benchmark_df = run_benchmark(portfolio_df, pw_slice, cfg.benchmark_ticker)

    merged = (
        portfolio_df[["rebal_date", "entry_date", "exit_date",
                       "portfolio_return", "n_held", "tickers_held"]]
        .merge(benchmark_df[["rebal_date", "benchmark_return"]], on="rebal_date", how="inner")
        .dropna(subset=["portfolio_return"])
    )
    merged["cum_portfolio"] = (1 + merged["portfolio_return"]).cumprod()
    merged["cum_benchmark"] = (1 + merged["benchmark_return"].fillna(0)).cumprod()

    os.makedirs(out_dir, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-")
    out_path   = os.path.join(out_dir, f"equity_{safe_label}.csv")
    merged.to_csv(out_path, sep=";", decimal=",", index=False)
    print(f"    Saved equity curve → {out_path}")


# =============================================================================
# MAIN WALK-FORWARD FUNCTION
# =============================================================================

def run_walk_forward(
    top_n_winners: int | None = None,
    param_grid: dict | None = None,
    save_results: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full Train → Test Walk-Forward Validation.

    Parameters
    ----------
    top_n_winners : int | None
        Number of top In-Sample configurations to evaluate Out-of-Sample.
        When None (default), calculated dynamically as:
            max(1, int(n_valid_trials * PORCENTAJE_TOP_GANADORES))
        i.e. the Top 10% of valid In-Sample results.
        Pass an explicit integer to override the percentage.
    param_grid : dict | None
        Grid to sweep. Defaults to DEFAULT_PARAM_GRID.
    save_results : bool
        Save comparison CSV and equity curves to results/.
    verbose : bool
        Print detailed progress.

    Returns
    -------
    pd.DataFrame — comparison table with Train + Test metrics per config.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    ensure_dirs()
    today_str = date.today().isoformat()

    train_range = (FECHA_INICIO_ANALISIS, FECHA_TRAIN_END)
    test_range  = (FECHA_TEST_START, today_str)

    if verbose:
        pct_label = (
            f"Top {int(PORCENTAJE_TOP_GANADORES * 100)}% (dynamic)"
            if top_n_winners is None
            else f"Top {top_n_winners} (override)"
        )
        print(f"\n{'='*70}")
        print("  WALK-FORWARD VALIDATION — Hybrid Momentum Strategy")
        print(f"{'='*70}")
        print(f"  Train (In-Sample) : {train_range[0]} → {train_range[1]}")
        print(f"  Test  (OOS)       : {test_range[0]}  → {test_range[1]}")
        print(f"  Winner selection  : {pct_label}")
        print(f"{'='*70}\n")

    # ── STEP 1: Load full data once ──────────────────────────────────────────
    df_clusters, price_wide = _load_full_data()

    # ── STEP 2: In-Sample Grid Search ────────────────────────────────────────
    if verbose:
        print("\n[STEP 1/3] Running In-Sample Grid Search...")

    train_csv = os.path.join(DIR_RESULTS, "grid_search_insample.csv")
    grid_df = run_grid_search(
        param_grid=param_grid,
        date_range=train_range,
        plot=False,
        save_csv=train_csv if save_results else None,
        df_clusters_preloaded=df_clusters,
        price_wide_preloaded=price_wide,
    )

    if grid_df.empty:
        print("  [ERROR] Grid search returned no valid results. Aborting.")
        return pd.DataFrame()

    # ── STEP 3: Select Top 10% In-Sample Winners (dynamic) ───────────────────
    param_keys = list(param_grid.keys())

    # Dynamic selection: Top 10% of valid trials, minimum 1
    if top_n_winners is None:
        n_ganadores = max(1, int(len(grid_df) * PORCENTAJE_TOP_GANADORES))
    else:
        n_ganadores = max(1, top_n_winners)

    top_configs = grid_df.head(n_ganadores)

    if verbose:
        print(f"\n[STEP 2/3] Selected Top {int(PORCENTAJE_TOP_GANADORES*100)}% "
              f"= {n_ganadores} configs from {len(grid_df)} valid In-Sample trials:")
        print(f"{'─'*70}")
        display_cols = param_keys + ["Sharpe", "CAGR (%)", "MaxDD (%)"]
        print(top_configs[display_cols].to_string(index=True))
        print(f"{'─'*70}\n")

    # ── STEP 4: Out-of-Sample Evaluation ─────────────────────────────────────
    if verbose:
        print(f"[STEP 3/3] Evaluating {len(top_configs)} configs Out-of-Sample...")

    equity_dir = os.path.join(DIR_RESULTS, "walk_forward_equity_curves")
    comparison_rows: list[dict] = []

    for rank, (_, row) in enumerate(top_configs.iterrows(), start=1):
        params = {k: row[k] for k in param_keys}

        sharpe_train = row.get("Sharpe",    np.nan)
        cagr_train   = row.get("CAGR (%)",  np.nan)
        maxdd_train  = row.get("MaxDD (%)", np.nan)

        if verbose:
            print(f"  [{rank:>2}/{len(top_configs)}] Testing: {params}")

        # Run OOS
        test_metrics = _run_single(params, test_range, df_clusters, price_wide)

        if test_metrics:
            sharpe_test = round(test_metrics.get("sharpe_portfolio", np.nan) or np.nan, 3)
            cagr_test   = round((test_metrics.get("cagr_portfolio",  np.nan) or np.nan) * 100, 2)
            maxdd_test  = round((test_metrics.get("maxdd_portfolio",  np.nan) or np.nan) * 100, 2)
        else:
            sharpe_test = cagr_test = maxdd_test = np.nan
            if verbose:
                print(f"       → No trades in test window (all-cash or no data)")

        degradacion = round(
            float(sharpe_train) - float(sharpe_test)
            if pd.notna(sharpe_train) and pd.notna(sharpe_test)
            else np.nan,
            3,
        )

        comp_row = {
            "rank_insample": rank,
            **params,
            "Sharpe_Train":       sharpe_train,
            "Sharpe_Test":        sharpe_test,
            "Degradacion_Sharpe": degradacion,
            "CAGR_Train (%)":     cagr_train,
            "CAGR_Test (%)":      cagr_test,
            "MaxDD_Train (%)":    maxdd_train,
            "MaxDD_Test (%)":     maxdd_test,
        }
        comparison_rows.append(comp_row)

        # Save equity curves for both periods
        if save_results:
            _save_equity_curve(params, train_range, df_clusters, price_wide,
                               equity_dir, f"rank{rank:02d}_train")
            _save_equity_curve(params, test_range,  df_clusters, price_wide,
                               equity_dir, f"rank{rank:02d}_test")

    comparison_df = pd.DataFrame(comparison_rows)

    # ── STEP 5: Print Summary ─────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print("  WALK-FORWARD COMPARISON — RESULTS")
        print(f"{'='*70}")
        summary_cols = (
            param_keys
            + ["Sharpe_Train", "Sharpe_Test", "Degradacion_Sharpe",
               "CAGR_Train (%)", "CAGR_Test (%)"]
        )
        print(comparison_df[summary_cols].to_string(index=False))
        print(f"{'='*70}")

        # Stats
        valid = comparison_df.dropna(subset=["Degradacion_Sharpe"])
        if not valid.empty:
            avg_deg = valid["Degradacion_Sharpe"].mean()
            pct_positive_test = (valid["Sharpe_Test"] > 0).mean() * 100
            print(f"\n  Avg Sharpe degradation   : {avg_deg:+.3f}")
            print(f"  % configs Sharpe_Test > 0: {pct_positive_test:.0f}%")

    # ── STEP 6: Save comparison CSV ───────────────────────────────────────────
    if save_results:
        out_csv = os.path.join(DIR_RESULTS, "walk_forward_comparison.csv")
        comparison_df.to_csv(out_csv, sep=";", decimal=",", index=False)
        print(f"\n  Walk-Forward results saved → {out_csv}")

    return comparison_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    wf_df = run_walk_forward(
        top_n_winners=10,
        param_grid=DEFAULT_PARAM_GRID,
        save_results=True,
        verbose=True,
    )
    print(f"\nWalk-Forward complete. {len(wf_df)} configurations compared.")
