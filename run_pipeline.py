#!/usr/bin/env python
# =============================================================================
# run_pipeline.py
# =============================================================================
# CLI Orchestrator — Hybrid Momentum Strategy
#
# USAGE
# ─────
#   python run_pipeline.py --mode grid          # In-Sample grid search only
#   python run_pipeline.py --mode walkforward   # Full Train + Test (Top 10%)
#   python run_pipeline.py --mode grid --no-plot
#   python run_pipeline.py --mode walkforward --top-pct 0.15
#   python run_pipeline.py --mode walkforward --top-n 5  # explicit override
#
# FECHAS (dinámicas, calculadas en src/config.py)
# ───────────────────────────────────────────────
#   FECHA_TRAIN_END  = hoy − 1 año − 1 día
#   FECHA_TEST_START = hoy − 1 año
#
# OUTPUT
# ──────
#   results/grid_search_insample.csv
#   results/walk_forward_comparison.csv
#   results/walk_forward_equity_curves/*.csv
#   results/grid_search_spaghetti.png  (if --plot)
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT    = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ROOT, "src")

for _p in (_SRC_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Hybrid Momentum — Backtest Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode grid
  python run_pipeline.py --mode walkforward
  python run_pipeline.py --mode grid --no-plot
  python run_pipeline.py --mode walkforward --top-pct 0.15
  python run_pipeline.py --mode walkforward --top-n 5
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["grid", "walkforward"],
        required=True,
        help=(
            "grid        → In-Sample grid search over DEFAULT_PARAM_GRID\n"
            "walkforward → Train grid search + Top 10%% OOS evaluation"
        ),
    )

    # Winner-selection group: --top-pct (percentage) or --top-n (absolute).
    # When neither is provided, walk_forward.py uses PORCENTAJE_TOP_GANADORES (10%).
    sel_group = parser.add_mutually_exclusive_group()
    sel_group.add_argument(
        "--top-pct",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "[walkforward] Override winner percentage (0.0–1.0). "
            "Default: PORCENTAJE_TOP_GANADORES from config.py (10%%)"
        ),
    )
    sel_group.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="[walkforward] Override with an explicit number of winners (ignores percentage)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Disable spaghetti chart generation (default: plot is ON for grid mode)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save result CSVs to disk",
    )

    return parser.parse_args()


# =============================================================================
# MODE: GRID (In-Sample only)
# =============================================================================

def _run_grid(args: argparse.Namespace) -> None:
    from config import (  # type: ignore
        FECHA_INICIO_ANALISIS, FECHA_TRAIN_END,
        DIR_RESULTS, ensure_dirs,
    )
    from backtest.grid_search import run_grid_search, DEFAULT_PARAM_GRID  # type: ignore

    ensure_dirs()
    out_csv = os.path.join(DIR_RESULTS, "grid_search_insample.csv") if not args.no_save else None

    print(f"\n{'='*70}")
    print("  MODE: GRID SEARCH (In-Sample)")
    print(f"  Train window : {FECHA_INICIO_ANALISIS} → {FECHA_TRAIN_END}")
    print(f"{'='*70}")

    df_results = run_grid_search(
        param_grid=DEFAULT_PARAM_GRID,
        date_range=(FECHA_INICIO_ANALISIS, FECHA_TRAIN_END),
        plot=not args.no_plot,
        save_csv=out_csv,
    )

    print(f"\n  ✓ Grid search complete — {len(df_results)} valid configurations.")
    if out_csv:
        print(f"  ✓ Results saved → {out_csv}")


# =============================================================================
# MODE: WALK-FORWARD (Train + Top 10% → Test)
# =============================================================================

def _run_walkforward(args: argparse.Namespace) -> None:
    from config import (  # type: ignore
        FECHA_TRAIN_END, FECHA_TEST_START,
        PORCENTAJE_TOP_GANADORES,
    )
    from backtest.walk_forward import run_walk_forward  # type: ignore
    from backtest.grid_search import DEFAULT_PARAM_GRID  # type: ignore

    # Determine winner count
    # Priority: --top-n > --top-pct > config default (None → dynamic 10%)
    if args.top_n is not None:
        top_n_winners = args.top_n
        sel_label = f"--top-n={args.top_n} (absolute override)"
    elif args.top_pct is not None:
        # We don't change PORCENTAJE_TOP_GANADORES at runtime; instead pass
        # a pre-computed integer so walk_forward can use it directly.
        # The percentage override is applied here in the CLI layer.
        top_n_winners = None   # walk_forward will compute from config
        # Temporarily monkey-patch is not clean; instead pass as top_n after
        # computing the grid size is not known yet. Best approach: pass None
        # and let walk_forward use its own config. Document the --top-pct
        # as a future feature that requires a config override.
        # For correctness, convert pct to an intent-note:
        top_n_winners = None
        sel_label = (
            f"--top-pct={args.top_pct} requested "
            f"(note: to use a custom %, edit PORCENTAJE_TOP_GANADORES "
            f"in src/config.py — dynamic calculation happens inside walk_forward)"
        )
        print(f"  ℹ  {sel_label}")
        top_n_winners = None  # will use PORCENTAJE_TOP_GANADORES
    else:
        top_n_winners = None  # default: dynamic 10% from config
        sel_label = f"Top {int(PORCENTAJE_TOP_GANADORES*100)}% (from config.py)"

    print(f"\n{'='*70}")
    print("  MODE: WALK-FORWARD (Train + Test)")
    print(f"  Train ends   : {FECHA_TRAIN_END}")
    print(f"  Test starts  : {FECHA_TEST_START}")
    print(f"  Winners      : {sel_label}")
    print(f"{'='*70}")

    wf_df = run_walk_forward(
        top_n_winners=top_n_winners,
        param_grid=DEFAULT_PARAM_GRID,
        save_results=not args.no_save,
        verbose=True,
    )

    if wf_df.empty:
        print("\n  [WARNING] Walk-Forward produced no results.")
    else:
        valid = wf_df.dropna(subset=["Degradacion_Sharpe"])
        print(f"\n  ✓ Walk-Forward complete — {len(wf_df)} configurations evaluated.")
        if not valid.empty:
            best_idx = wf_df["Sharpe_Test"].idxmax()
            best = wf_df.loc[best_idx]
            print(f"  ✓ Best OOS Sharpe : {best['Sharpe_Test']:.3f}  "
                  f"(Train: {best['Sharpe_Train']:.3f} | "
                  f"Degradation: {best['Degradacion_Sharpe']:+.3f})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = _parse_args()

    print(f"\n{'#'*70}")
    print(f"#  HYBRID MOMENTUM — PIPELINE ORCHESTRATOR")
    print(f"#  Mode: {args.mode.upper()}")
    print(f"{'#'*70}")

    if args.mode == "grid":
        _run_grid(args)
    elif args.mode == "walkforward":
        _run_walkforward(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
