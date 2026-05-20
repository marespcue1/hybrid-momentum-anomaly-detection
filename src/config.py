# =============================================================================
# src/config.py
# =============================================================================
# Single source of truth for the entire pipeline.
# All paths, dates and constants live here — never hard-coded elsewhere.
#
# FECHAS DINÁMICAS
# ────────────────
#   FECHA_INICIO_ANALISIS → fijo: "2017-01-01"
#   FECHA_TEST_START      → dinámico: exactamente 1 año antes de hoy
#   FECHA_TRAIN_END       → dinámico: 1 día antes de FECHA_TEST_START
#
#   Las tres se exportan como strings "YYYY-MM-DD" para no romper ningún
#   consumer que haga pd.Timestamp(FECHA_TEST_START) o comparaciones de texto.
# =============================================================================

import os
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file so it works from any cwd)
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# =============================================================================
# WALK-FORWARD DATE SPLITS  (all exported as "YYYY-MM-DD" strings)
# =============================================================================
FECHA_INICIO_ANALISIS: str = "2017-01-01"   # Fixed: start of full history

# Dynamic: Test window starts exactly 1 year before today
_test_start: date  = date.today().replace(year=date.today().year - 1)
FECHA_TEST_START: str = _test_start.strftime("%Y-%m-%d")

# Dynamic: Train ends the day before Test starts
_train_end: date   = _test_start - timedelta(days=1)
FECHA_TRAIN_END: str  = _train_end.strftime("%Y-%m-%d")

# =============================================================================
# WALK-FORWARD SELECTION PERCENTAGE
# =============================================================================
# Top fraction of In-Sample trials to evaluate Out-of-Sample.
# Formula: n_ganadores = max(1, int(n_trials * PORCENTAJE_TOP_GANADORES))
PORCENTAJE_TOP_GANADORES: float = 0.10   # 10 %

# =============================================================================
# RUTAS DE DATOS DE ENTRADA
# =============================================================================
FILE_CLUSTERS: str = os.path.join(_PROJECT_ROOT, "data", "processed", "05_dataset_clusters.csv")
FILE_PRICES:   str = os.path.join(_PROJECT_ROOT, "data", "raw",       "01_dataset_bruto.csv")

# =============================================================================
# RUTAS DE RESULTADOS
# =============================================================================
DIR_RESULTS: str  = os.path.join(_PROJECT_ROOT, "results")
DIR_FIGURES: str  = os.path.join(_PROJECT_ROOT, "results", "figures")

# Ficheros de salida del pipeline
# ─────────────────────────────────────────────────────────────────────────────
# FILE_GRID_INSAMPLE  → resultados del Grid Search In-Sample completo (12.800 filas)
#                       generado por walk_forward.py en el Paso 1.
# FILE_WF_COMPARISON  → tabla comparativa Train/Test de los ganadores Top 10 %
#                       generada por walk_forward.py en el Paso 5.
# Los dos nombres legacy (FILE_GRID_RESULTS / FILE_WF_RESULTS) se mantienen
# como alias para no romper imports existentes.
FILE_GRID_INSAMPLE:  str = os.path.join(DIR_RESULTS, "grid_search_insample.csv")
FILE_WF_COMPARISON:  str = os.path.join(DIR_RESULTS, "walk_forward_comparison.csv")

# Legacy aliases (apuntan a los mismos archivos canónicos)
FILE_GRID_RESULTS: str = FILE_GRID_INSAMPLE
FILE_WF_RESULTS:   str = FILE_WF_COMPARISON
FILE_WF_EQUITY:    str = os.path.join(DIR_RESULTS, "walk_forward_equity_curves.csv")

# =============================================================================
# BENCHMARK
# =============================================================================
BENCHMARK_TICKER: str = "SPY"


def ensure_dirs() -> None:
    """Crea los directorios de resultados si no existen todavía."""
    for d in (DIR_RESULTS, DIR_FIGURES):
        os.makedirs(d, exist_ok=True)
