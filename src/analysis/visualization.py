# =============================================================================
# src/analysis/visualization.py
# =============================================================================

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuración de rutas ────────────────────────────────────────────────────
_THIS_FILE    = os.path.abspath(__file__)
_ANALYSIS_DIR = os.path.dirname(_THIS_FILE)
_SRC_DIR      = os.path.dirname(_ANALYSIS_DIR)
_ROOT         = os.path.dirname(_SRC_DIR)

for _p in (_SRC_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # type: ignore
    DIR_RESULTS,
    DIR_FIGURES,
    FECHA_INICIO_ANALISIS,
    FECHA_TRAIN_END,
    FECHA_TEST_START,
    FILE_PRICES,
    FILE_GRID_INSAMPLE,
    FILE_WF_COMPARISON,
    ensure_dirs,
)

# Alias de compatibilidad para código que importe los nombres antiguos
FILE_GRID_RESULTS = FILE_GRID_INSAMPLE
FILE_WF_RESULTS   = FILE_WF_COMPARISON


# =============================================================================
# SECCIÓN 0 — CACHÉ INTERNA DE PRECIOS DIARIOS
# =============================================================================

_PRICE_WIDE_CACHE: pd.DataFrame | None = None


def _get_price_wide() -> pd.DataFrame:
    """Carga y almacena en caché la tabla de precios diarios en formato ancho.

    Único punto del módulo donde se lee FILE_PRICES. Todas las funciones de
    post-procesado utilizan esta caché para evitar lecturas redundantes de disco.

    Retorna
    -------
    pd.DataFrame  índice=Fecha (diario), columnas=tickers, valores=Precio_Adj.
    DataFrame vacío si el archivo no existe.
    """
    global _PRICE_WIDE_CACHE
    if _PRICE_WIDE_CACHE is not None:
        return _PRICE_WIDE_CACHE

    if not os.path.exists(FILE_PRICES):
        print(f"[visualization] Archivo de precios no encontrado: {FILE_PRICES}")
        return pd.DataFrame()

    df = pd.read_csv(
        FILE_PRICES, sep=";", decimal=",", low_memory=False,
        usecols=["Fecha", "Ticker", "Precio_Adj"],
    )
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    price_wide = (
        df[["Fecha", "Ticker", "Precio_Adj"]]
        .drop_duplicates(["Fecha", "Ticker"])
        .pivot(index="Fecha", columns="Ticker", values="Precio_Adj")
        .sort_index()
    )
    price_wide.columns.name = None
    _PRICE_WIDE_CACHE = price_wide
    return _PRICE_WIDE_CACHE


# =============================================================================
# SECCIÓN 1 — CARGADORES DE DATOS
# =============================================================================

def load_walk_forward_results(path: str | None = None) -> pd.DataFrame:
    """Carga la tabla comparativa Train/Test de los ganadores Walk-Forward.

    Archivo canónico: walk_forward_comparison.csv
    Columnas relevantes:
        rank_insample | params... | Sharpe_Train | Sharpe_Test
        Degradacion_Sharpe | CAGR_Train (%) | CAGR_Test (%)
        MaxDD_Train (%) | MaxDD_Test (%)

    Retorna DataFrame vacío si el archivo no existe todavía.
    """
    if path is None:
        path = FILE_WF_COMPARISON
    if not os.path.exists(path):
        print(f"[visualization] Archivo no encontrado: {path}")
        print("  -> Ejecutar primero: python run_pipeline.py --mode walkforward")
        return pd.DataFrame()
    return pd.read_csv(path, sep=";", decimal=",")


def load_grid_results(path: str | None = None) -> pd.DataFrame:
    """Carga los resultados completos del Grid Search In-Sample (12.800 filas).

    Archivo canónico: grid_search_insample.csv
    Columnas relevantes:
        params... | Sharpe | CAGR (%) | MaxDD (%) | HitRate (%)

    Es la fuente estadísticamente significativa para la distribución de
    hiperparámetros (Sección 5): N=12.800 combinaciones.
    """
    if path is None:
        path = FILE_GRID_INSAMPLE
    if not os.path.exists(path):
        print(f"[visualization] Archivo no encontrado: {path}")
        print("  -> Ejecutar primero: python run_pipeline.py --mode walkforward")
        return pd.DataFrame()
    return pd.read_csv(path, sep=";", decimal=",")


def load_daily_spy(
    start: str | None = None,
    end:   str | None = None,
    ticker: str = "SPY",
) -> pd.Series:
    """Carga los precios diarios del benchmark y los normaliza a base 1.0.

    Proporciona la curva de referencia diaria para superponerla en las
    gráficas de equidad. Los precios se extraen del archivo bruto gracias
    a la caché interna, sin lectura adicional de disco.

    Retorna serie vacía si el archivo no existe o el ticker no está presente.
    """
    price_wide = _get_price_wide()
    if price_wide.empty or ticker not in price_wide.columns:
        return pd.Series(dtype=float)

    spy = price_wide[ticker].dropna()
    if start:
        spy = spy[spy.index >= pd.Timestamp(start)]
    if end:
        spy = spy[spy.index <= pd.Timestamp(end)]
    if spy.empty:
        return pd.Series(dtype=float)

    return spy / spy.iloc[0]


def _run_engine_for_config(
    row: pd.Series,
    date_range: tuple[str, str],
) -> pd.DataFrame:
    """Re-ejecuta el motor de backtest para una configuración ganadora.

    Acepta tanto pd.Series (iterrows) como named tuple (itertuples).
    El archivo engine.py NO se modifica; únicamente se invoca con los
    parámetros almacenados en walk_forward_comparison.csv.

    Su único propósito es obtener el portfolio_df con las columnas
    entry_date, exit_date, tickers_held y weights_used, necesarias para
    la reconstrucción diaria mark-to-market.

    Retorna DataFrame vacío si la simulación no produce operaciones.
    """
    from backtest.engine import BacktestConfig, run  # type: ignore

    # Columnas que NO son parámetros del motor
    columnas_no_param = {
        "rank_insample", "rank_train", "config",
        "Sharpe_Train", "Sharpe_Test", "Degradacion_Sharpe", "Sharpe_Degradation_%",
        "CAGR_Train (%)", "CAGR_Test (%)", "CAGR_Train_%", "CAGR_Test_%",
        "MaxDD_Train (%)", "MaxDD_Test (%)", "MaxDD_Train_%", "MaxDD_Test_%",
        "Sharpe", "CAGR (%)", "MaxDD (%)", "HitRate (%)",
        "ActivePeriods", "CashPeriods", "TotalPeriods", "BenchCAGR (%)", "BenchSharpe",
    }

    # Soporte para Series e itertuples (named tuple)
    if isinstance(row, pd.Series):
        param_keys = [c for c in row.index if c not in columnas_no_param]
        get_val = lambda k: row[k]
    else:
        # named tuple de itertuples
        param_keys = [f for f in row._fields if f not in columnas_no_param]
        get_val = lambda k: getattr(row, k)

    kwargs: dict = {}
    for k in param_keys:
        val = get_val(k)
        if pd.isna(val):
            continue
        if k in ("top_n", "anomaly_lookback_days", "active_cluster", "min_portfolio_size"):
            kwargs[k] = int(val)
        elif k == "rank_weighted":
            kwargs[k] = bool(val)
        elif k in ("rebalance_freq", "momentum_signal"):
            kwargs[k] = str(val)

    cfg = BacktestConfig(**kwargs, date_range=date_range, verbose=False, plot=False)
    result = run(cfg)
    return result.get("portfolio_df", pd.DataFrame())


def _top10pct_by_sharpe_IS(df_grid: pd.DataFrame) -> pd.DataFrame:
    """Filtra el Grid Search completo para retener el Top 10 % por Sharpe (IS).

    Fuente: grid_search_insample.csv (N=12.800 combinaciones).
    Columna de ranking: 'Sharpe' (métrica In-Sample).
    Pool premium resultante: ≈ 1.280 estrategias.

    CONTRATO DE ÍNDICE
    ──────────────────
    El DataFrame devuelto tiene SIEMPRE un índice entero secuencial
    reseteado (0, 1, 2, …) para evitar desalineaciones posicionales.

    Parámetros
    ----------
    df_grid : pd.DataFrame — tabla grid_search_insample.csv

    Retorna
    -------
    pd.DataFrame — Top 10 % con índice reseteado 0-based.
    """
    col = "Sharpe"
    if col not in df_grid.columns:
        print(f"[visualization] Columna '{col}' no encontrada en el Grid Search.")
        return df_grid.reset_index(drop=True)

    valido = df_grid.dropna(subset=[col])
    if valido.empty:
        return df_grid.reset_index(drop=True)

    umbral = valido[col].quantile(0.90)
    elite  = valido[valido[col] >= umbral].copy()

    if elite.empty:
        return df_grid.reset_index(drop=True)

    print(f"  [_top10pct_by_sharpe_IS] Umbral Sharpe >= {umbral:.4f}: {len(elite)}/{len(df_grid)} estrategias retenidas.")
    return elite.reset_index(drop=True)


def _top10pct_by_sharpe_test(df_wf: pd.DataFrame) -> pd.DataFrame:
    """Filtra el DataFrame de ganadores para retener el Top 10 % por Sharpe_Test.

    Fuente: walk_forward_comparison.csv.
    Columna de ranking: 'Sharpe_Test' (métrica OOS).

    CONTRATO DE ÍNDICE
    ──────────────────
    El DataFrame devuelto tiene SIEMPRE un índice entero secuencial
    reseteado (0, 1, 2, …).

    Parámetros
    ----------
    df_wf : pd.DataFrame — tabla walk_forward_comparison.csv

    Retorna
    -------
    pd.DataFrame — subconjunto Top 10 % con índice reseteado 0-based.
    """
    col = "Sharpe_Test"
    if col not in df_wf.columns:
        return df_wf.reset_index(drop=True)

    valido = df_wf.dropna(subset=[col])
    if valido.empty:
        return df_wf.reset_index(drop=True)

    umbral = valido[col].quantile(0.90)
    elite  = valido[valido[col] >= umbral].copy()

    if elite.empty:
        return df_wf.reset_index(drop=True)

    print(f"  [_top10pct_by_sharpe_test] Umbral Sharpe_Test >= {umbral:.4f}: {len(elite)}/{len(df_wf)} estrategias retenidas.")
    return elite.reset_index(drop=True)


# =============================================================================
# SECCIÓN 2 — MARK-TO-MARKET DIARIO REAL
# =============================================================================

def reconstruct_daily_equity(
    portfolio_df: pd.DataFrame,
    price_wide: pd.DataFrame | None = None,
) -> pd.Series:
    """Reconstruye la curva de equidad diaria con oscilación real de mercado.

    Función de post-procesado puro: NO ejecuta el motor, NO modifica engine.py
    y NO altera ninguna decisión de la estrategia.

    Algoritmo (por período i = [entry_date_i, exit_date_i])
    ────────────────────────────────────────────────────────
    Para cada día hábil d en [entry_date_i, exit_date_i]:

        valor(d) = Σ_k  w_k · P_k(d) / P_k(entry_date_i)

    donde w_k = peso del activo k y P_k = precio ajustado diario.

    El nivel acumulado se encadena entre períodos con el retorno oficial del
    motor para garantizar coherencia con el backtest original.
    Los períodos en efectivo (sin posiciones) generan segmentos planos.

    Retorna serie vacía si faltan datos o el portfolio_df está vacío.
    """
    if portfolio_df is None or portfolio_df.empty:
        return pd.Series(dtype=float)

    if price_wide is None:
        price_wide = _get_price_wide()
    if price_wide is None or price_wide.empty:
        return pd.Series(dtype=float)

    df = portfolio_df.copy()
    for col in ("entry_date", "exit_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    daily_dates:  list[pd.Timestamp] = []
    daily_values: list[float]        = []
    nivel_actual  = 1.0

    for _, fila in df.iterrows():
        entry_date = fila.get("entry_date")
        exit_date  = fila.get("exit_date")

        if pd.isna(entry_date) or pd.isna(exit_date):
            continue

        mascara       = (price_wide.index >= entry_date) & (price_wide.index <= exit_date)
        fechas_period = price_wide.index[mascara]

        if len(fechas_period) == 0:
            continue

        tickers = fila.get("tickers_held", [])
        weights = fila.get("weights_used",  [])

        if isinstance(tickers, str):
            import ast
            try:
                tickers = ast.literal_eval(tickers)
            except Exception:
                tickers = []
        if isinstance(weights, str):
            import ast
            try:
                weights = ast.literal_eval(weights)
            except Exception:
                weights = []

        # Período en efectivo: segmento plano
        if not tickers or not weights or len(tickers) != len(weights):
            for d in fechas_period:
                daily_dates.append(d)
                daily_values.append(nivel_actual)
            ret = fila.get("portfolio_return", 0.0)
            nivel_actual *= (1.0 + (ret if not pd.isna(ret) else 0.0))
            continue

        pares_validos = [(t, w) for t, w in zip(tickers, weights)
                         if t in price_wide.columns]
        if not pares_validos:
            for d in fechas_period:
                daily_dates.append(d)
                daily_values.append(nivel_actual)
            ret = fila.get("portfolio_return", 0.0)
            nivel_actual *= (1.0 + (ret if not pd.isna(ret) else 0.0))
            continue

        tickers_val = [p[0] for p in pares_validos]
        pesos_val   = np.array([p[1] for p in pares_validos], dtype=float)
        pesos_val  /= pesos_val.sum()

        precios_entrada = price_wide.loc[entry_date, tickers_val]

        for d in fechas_period:
            precios_hoy   = price_wide.loc[d, tickers_val]
            ratios        = (precios_hoy / precios_entrada).fillna(1.0).values
            daily_dates.append(d)
            daily_values.append(nivel_actual * float(np.dot(pesos_val, ratios)))

        ret = fila.get("portfolio_return", 0.0)
        nivel_actual *= (1.0 + (ret if not pd.isna(ret) else 0.0))

    if not daily_dates:
        return pd.Series(dtype=float)

    serie = pd.Series(daily_values, index=pd.DatetimeIndex(daily_dates))
    return serie[~serie.index.duplicated(keep="last")].sort_index()


# =============================================================================
# SECCIÓN 3 — CURVA DE EQUIDAD CON LÍNEA DIVISORIA TRAIN/TEST
# =============================================================================

def plot_equity_curve_with_split(
    rank: int = 1,
    wf_results_df: pd.DataFrame | None = None,
    ax=None,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """Representa la curva de equidad diaria real de la configuración ganadora.

    La curva se dibuja con resolución diaria usando reconstruct_daily_equity().
    No se emplea ffill() ni relleno plano: la curva oscila día a día con los
    precios reales de los activos en cartera.

    Se incluye únicamente la línea vertical punteada que separa los períodos
    Train y Test. No aparecen títulos ni textos flotantes interiores.

    FUENTE DE DATOS
    ───────────────
    walk_forward_comparison.csv — columna 'rank_insample'.
    La búsqueda se realiza por valor de columna (no índice posicional).

    Parámetros
    ----------
    rank          : rango del ganador (1 = mejor Sharpe en Train).
    wf_results_df : tabla walk-forward (carga automática si es None).
    ax            : ejes de matplotlib (crea figura nueva si es None).
    save          : guarda PNG en results/figures/ si es True.
    filename      : nombre del archivo de salida (opcional).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib no disponible.")
        return

    if wf_results_df is None:
        wf_results_df = load_walk_forward_results()
    if wf_results_df.empty:
        return

    # Determinar columna de rank (compatibilidad con ambos formatos CSV)
    rank_col = "rank_insample" if "rank_insample" in wf_results_df.columns else "rank_train"
    if rank_col not in wf_results_df.columns:
        print("[visualization] Columna de rank no encontrada ('rank_insample' / 'rank_train').")
        return

    mascara_rank = wf_results_df[rank_col] == rank
    if not mascara_rank.any():
        print(f"[visualization] Rango #{rank} no encontrado en '{rank_col}'.")
        return

    # Verificar duplicados de rank (debería haber exactamente 1 fila)
    n_coincidencias = int(mascara_rank.sum())
    if n_coincidencias > 1:
        print(f"[visualization] AVISO: Rango #{rank} aparece {n_coincidencias} veces; usando la primera.")

    fila_ganadora = wf_results_df.loc[mascara_rank].iloc[0]

    # Validación: confirmar que el valor de rank coincide con lo solicitado
    rank_real = fila_ganadora[rank_col]
    if rank_real != rank:
        print(f"[visualization] ERROR: Rank mismatch — solicitado #{rank}, obtenido #{rank_real}.")
        return

    price_wide = _get_price_wide()
    if price_wide.empty:
        return

    train_range = (FECHA_INICIO_ANALISIS, FECHA_TRAIN_END)
    test_range  = (FECHA_TEST_START, price_wide.index.max().strftime("%Y-%m-%d"))

    pf_train = _run_engine_for_config(fila_ganadora, train_range)
    pf_test  = _run_engine_for_config(fila_ganadora, test_range)

    d_train = reconstruct_daily_equity(pf_train, price_wide)
    d_test  = pd.Series(dtype=float)

    if not pf_test.empty:
        d_test_raw = reconstruct_daily_equity(pf_test, price_wide)
        if not d_test_raw.empty:
            nivel_fin  = d_train.iloc[-1] if not d_train.empty else 1.0
            d_test     = d_test_raw * nivel_fin

    if not d_train.empty and not d_test.empty:
        equidad = pd.concat([d_train, d_test])
    elif not d_train.empty:
        equidad = d_train
    else:
        equidad = d_test

    equidad = equidad[~equidad.index.duplicated(keep="last")].sort_index()
    if equidad.empty:
        return

    spy = load_daily_spy(
        start=equidad.index.min().strftime("%Y-%m-%d"),
        end=equidad.index.max().strftime("%Y-%m-%d"),
    )
    spy = spy * equidad.iloc[0]

    figura_propia = ax is None
    if figura_propia:
        fig, ax = plt.subplots(figsize=(14, 6))

    fecha_corte = pd.Timestamp(FECHA_TEST_START)

    ax.plot(equidad.index, equidad.values,
            color="#1f4e79", linewidth=2.5,
            label="Hybrid Momentum (mark-to-market diario)", zorder=5)

    if not spy.empty:
        ax.plot(spy.index, spy.values,
                color="#cc0000", linewidth=1.3, alpha=0.75,
                label="SPY (diario)", zorder=4)

    ax.axvline(fecha_corte, color="black", linestyle=":", linewidth=2.0,
               label=f"Corte ({FECHA_TEST_START})", zorder=6)

    ax.axvspan(equidad.index.min(), fecha_corte,
               alpha=0.04, color="#1f4e79")
    if not d_test.empty:
        ax.axvspan(fecha_corte, equidad.index.max(),
                   alpha=0.06, color="#ff7f0e")

    ax.set_ylabel("Retorno Acumulado (base = 1.0)", fontsize=10)
    ax.set_xlabel("Fecha", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figura_propia:
        plt.tight_layout()
        if save:
            ensure_dirs()
            nombre = filename or f"equity_curve_rank{rank:02d}_daily.png"
            salida = os.path.join(DIR_FIGURES, nombre)
            plt.savefig(salida, dpi=150, bbox_inches="tight")
            print(f"  [visualization] Guardado -> {salida}")
        plt.show()


# =============================================================================
# SECCIÓN 4 — SCATTER DE DEGRADACIÓN DE SHARPE
# =============================================================================

def plot_sharpe_scatter(
    wf_df: pd.DataFrame | None = None,
    ax=None,
    save: bool = False,
    filename: str = "sharpe_scatter.png",
) -> None:
    """Diagrama de dispersión: Sharpe Train vs Sharpe Test de los ganadores.

    Los puntos por encima de la diagonal indican rendimiento OOS superior
    al IS (overfitting evitado). Los puntos por debajo muestran degradación.
    El color codifica el porcentaje de degradación del Sharpe.
    Sin título ni textos flotantes interiores.

    FUENTE DE DATOS: walk_forward_comparison.csv
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[visualization] matplotlib no disponible.")
        return

    if wf_df is None:
        wf_df = load_walk_forward_results()
    if wf_df.empty:
        return

    valido = wf_df.dropna(subset=["Sharpe_Train", "Sharpe_Test"]).copy()
    if valido.empty:
        return

    figura_propia = ax is None
    if figura_propia:
        fig, ax = plt.subplots(figsize=(9, 7))

    # Columna de degradación: compatibilidad con ambos formatos
    deg_col = (
        "Degradacion_Sharpe" if "Degradacion_Sharpe" in valido.columns
        else "Sharpe_Degradation_%" if "Sharpe_Degradation_%" in valido.columns
        else None
    )
    deg = valido[deg_col].fillna(0) if deg_col else pd.Series(0, index=valido.index)

    norma   = plt.Normalize(deg.min(), deg.max())
    colores = cm.RdYlGn(norma(deg.values))  # type: ignore[attr-defined]

    lim_min = min(valido["Sharpe_Train"].min(), valido["Sharpe_Test"].min()) - 0.1
    lim_max = max(valido["Sharpe_Train"].max(), valido["Sharpe_Test"].max()) + 0.1

    ax.fill_between([lim_min, lim_max], [lim_min, lim_max], lim_max,
                    alpha=0.05, color="green")
    ax.fill_between([lim_min, lim_max], lim_min, [lim_min, lim_max],
                    alpha=0.05, color="red")

    ax.scatter(valido["Sharpe_Train"], valido["Sharpe_Test"],
               c=colores, s=130, edgecolors="white", linewidths=1.0, zorder=5)

    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "k--", linewidth=1.2, alpha=0.6, label="Sin degradación (Train = Test)")
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.4)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norma, cmap="RdYlGn"),  # type: ignore[attr-defined]
        ax=ax, shrink=0.8,
    )
    cbar.set_label("Degradación Sharpe", fontsize=9)

    ax.set_xlabel("Sharpe Train (In-Sample)", fontsize=11)
    ax.set_ylabel("Sharpe Test (Out-of-Sample)", fontsize=11)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figura_propia:
        plt.tight_layout()
        if save:
            ensure_dirs()
            salida = os.path.join(DIR_FIGURES, filename)
            plt.savefig(salida, dpi=150, bbox_inches="tight")
            print(f"  [visualization] Guardado -> {salida}")
        plt.show()


# =============================================================================
# SECCIÓN 5 — DISTRIBUCIÓN DE HIPERPARÁMETROS (TOP 10 % IS POR Sharpe)
# =============================================================================

def plot_hyperparameter_distribution(
    wf_df: pd.DataFrame | None = None,
    param_keys: list[str] | None = None,
    save: bool = False,
    filename: str = "hyperparameter_distribution.png",
) -> None:
    """Distribución de hiperparámetros en el Top 10 % del Grid Search IS.

    FUENTE DE DATOS (v3)
    ────────────────────
    Se carga grid_search_insample.csv (N=12.800 combinaciones) y se filtra
    el Top 10 % por columna 'Sharpe' (In-Sample). El pool premium resultante
    es ≈ 1.280 estrategias, estadísticamente representativo del espacio de
    búsqueda completo.

    El parámetro `wf_df` se mantiene por compatibilidad de firma pero se
    ignora: la función siempre opera sobre el Grid Search completo IS.
    Si grid_search_insample.csv no existe, cae al comportamiento anterior
    usando wf_df como fallback.

    COHERENCIA MATEMÁTICA
    ─────────────────────
    La suma de barras de cada subgráfico debe igualar len(elite_reset).
    Se emite un [AVISO] si la suma difiere (NaN en ese parámetro).

    No se incluye título superior en la figura.

    Parámetros
    ----------
    wf_df      : ignorado (se conserva por compatibilidad de firma).
    param_keys : columnas de hiperparámetros a analizar (auto-detectadas si None).
    save       : guarda PNG en results/figures/ si es True.
    filename   : nombre del archivo de salida.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib no disponible.")
        return

    # ── Cargar Grid Search IS completo (fuente canónica) ─────────────────────
    df_grid = load_grid_results()

    if df_grid.empty:
        # Fallback: si no existe el grid IS, usar wf_df
        if wf_df is None:
            wf_df = load_walk_forward_results()
        if wf_df is None or wf_df.empty:
            print("[visualization] Sin datos disponibles para Sección 5.")
            return
        df_grid = wf_df
        print("[visualization] Usando walk_forward_comparison.csv como fallback.")

    # ── Top 10 % por Sharpe IS (≈ 1.280 estrategias) ─────────────────────────
    # Si la columna 'Sharpe' existe -> Grid IS. Si no -> fallback con Sharpe_Train.
    if "Sharpe" in df_grid.columns:
        elite_reset = _top10pct_by_sharpe_IS(df_grid)
        fuente_label = "Top 10 % Grid IS por Sharpe (In-Sample)"
    else:
        elite_reset = _top10pct_by_sharpe_test(df_grid)
        fuente_label = "Top 10 % por Sharpe_Test (OOS) — fallback"

    n_elite = len(elite_reset)
    print(f"[visualization] Sección 5 — {fuente_label}: N={n_elite}")

    # ── Auto-detectar columnas de parámetros ─────────────────────────────────
    if param_keys is None:
        columnas_metricas = {
            "rank_insample", "rank_train", "config",
            "Sharpe_Train", "Sharpe_Test", "Degradacion_Sharpe", "Sharpe_Degradation_%",
            "CAGR_Train (%)", "CAGR_Test (%)", "CAGR_Train_%", "CAGR_Test_%",
            "MaxDD_Train (%)", "MaxDD_Test (%)", "MaxDD_Train_%", "MaxDD_Test_%",
            "Sharpe", "CAGR (%)", "MaxDD (%)", "HitRate (%)",
            "ActivePeriods", "CashPeriods", "TotalPeriods", "BenchCAGR (%)", "BenchSharpe",
        }
        param_keys = [c for c in elite_reset.columns if c not in columnas_metricas]

    if not param_keys:
        print("[visualization] No se encontraron columnas de hiperparámetros.")
        return

    COLORES_CLUSTER = {
        0: "#e63946", 1: "#f4a261", 2: "#2a9d8f", 3: "#457b9d", 4: "#9d0208"
    }

    ncols = min(4, len(param_keys))
    nrows = (len(param_keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    ejes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    # Tracking de coherencia por parámetro
    coherencia_report: list[dict] = []

    for i, clave in enumerate(param_keys):
        ax = ejes_flat[i]
        if clave not in elite_reset.columns:
            ax.set_visible(False)
            coherencia_report.append({"param": clave, "suma": "N/A", "n_elite": n_elite, "ok": False})
            continue

        conteos = elite_reset[clave].value_counts().sort_index()
        nan_count = elite_reset[clave].isna().sum()

        # Verificación de coherencia: suma de conteos + NaN debe igualar n_elite
        suma = conteos.sum()
        ok = (suma + nan_count) == n_elite
        coherencia_report.append({
            "param": clave, "suma": int(suma), "nan": int(nan_count),
            "n_elite": n_elite, "ok": ok,
        })

        if not ok:
            print(
                f"  [AVISO] '{clave}': conteos({suma}) + NaN({nan_count}) ≠ n_elite({n_elite})."
            )

        if clave == "active_cluster":
            colores_barras = [COLORES_CLUSTER.get(int(k), "#888") for k in conteos.index]
        else:
            colores_barras = "#457b9d"

        barras = ax.bar(range(len(conteos)), conteos.values,
                        color=colores_barras, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(conteos)))
        ax.set_xticklabels([str(v) for v in conteos.index],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Nº configuraciones", fontsize=8)
        ax.set_xlabel(clave, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for barra, val in zip(barras, conteos.values):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + max(0.3, n_elite * 0.005),
                    str(val), ha="center", va="bottom", fontsize=8)

    for j in range(len(param_keys), len(ejes_flat)):
        ejes_flat[j].set_visible(False)

    # Resumen de coherencia matemática
    print(f"[visualization] Sección 5 — Coherencia de conteos (n_elite={n_elite}):")
    all_ok = True
    for entry in coherencia_report:
        status = "OK" if entry["ok"] else "WARN"
        if "nan" in entry:
            print(f"  [{status}] {entry['param']}: conteos={entry['suma']}, NaN={entry['nan']}, total={entry['suma'] + entry['nan']}")
        else:
            print(f"  [{status}] {entry['param']}: columna no encontrada")
        if not entry["ok"]:
            all_ok = False
    if all_ok:
        print("  -> Todos los parámetros son matemáticamente coherentes.")

    plt.tight_layout()

    if save:
        ensure_dirs()
        salida = os.path.join(DIR_FIGURES, filename)
        plt.savefig(salida, dpi=150, bbox_inches="tight")
        print(f"  [visualization] Guardado -> {salida}  ({fuente_label}, N={n_elite})")
    plt.show()


# =============================================================================
# SECCIÓN 6 — BARRAS DE DEGRADACIÓN DE SHARPE POR GANADOR
# =============================================================================

def plot_degradation_bar(
    wf_df: pd.DataFrame | None = None,
    ax=None,
    save: bool = False,
    filename: str = "degradation_bar.png",
) -> None:
    """Barras horizontales comparando Sharpe Train y Sharpe Test por ganador.

    Muestra el Sharpe In-Sample (azul) y Out-of-Sample (naranja) en barras
    paralelas, ordenadas de menor a mayor Sharpe Train.

    FUENTE DE DATOS: walk_forward_comparison.csv
    Etiquetas del eje Y tomadas de 'rank_insample' (valor real de datos).

    Sin título superior ni textos flotantes interiores.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib no disponible.")
        return

    if wf_df is None:
        wf_df = load_walk_forward_results()
    if wf_df.empty:
        return

    # Columna de rank (compatibilidad ambos formatos)
    rank_col = (
        "rank_insample" if "rank_insample" in wf_df.columns
        else "rank_train" if "rank_train" in wf_df.columns
        else None
    )
    if rank_col is None:
        print("[visualization] Columna de rank no encontrada.")
        return

    valido = wf_df.dropna(subset=["Sharpe_Train", "Sharpe_Test"]).copy()
    valido = valido.sort_values("Sharpe_Train", ascending=True).reset_index(drop=True)
    if valido.empty:
        return

    # Limitar a un número manejable de barras para legibilidad
    MAX_BARS = 50
    n_total_validos = len(valido)
    if n_total_validos > MAX_BARS:
        print(f"[visualization] Sección 6 — {n_total_validos} configs válidas; mostrando top/bottom {MAX_BARS//2} por Sharpe_Train.")
        # Mostrar las primeras MAX_BARS//2 (menor Sharpe_Train) y las últimas MAX_BARS//2 (mayor Sharpe_Train)
        n_half = MAX_BARS // 2
        valido_display = pd.concat([valido.iloc[:n_half], valido.iloc[-n_half:]]).reset_index(drop=True)
    else:
        valido_display = valido

    figura_propia = ax is None
    if figura_propia:
        fig, ax = plt.subplots(figsize=(9, max(4, len(valido_display) * 0.5)))

    y    = np.arange(len(valido_display))
    alto = 0.35

    etiq = [f"#{int(r)}" for r in valido_display[rank_col]]

    ax.barh(y + alto / 2, valido_display["Sharpe_Train"], alto,
            label="Sharpe Train (IS)",  color="#1f4e79", alpha=0.85)
    ax.barh(y - alto / 2, valido_display["Sharpe_Test"].fillna(0), alto,
            label="Sharpe Test (OOS)", color="#ff7f0e", alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(1, color="green", linewidth=0.8, linestyle=":", alpha=0.7,
               label="Sharpe = 1.0")
    ax.set_yticks(y)
    ax.set_yticklabels(etiq, fontsize=9)
    ax.set_xlabel("Sharpe Ratio (anualizado)", fontsize=11)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if figura_propia:
        plt.tight_layout()
        if save:
            ensure_dirs()
            salida = os.path.join(DIR_FIGURES, filename)
            plt.savefig(salida, dpi=150, bbox_inches="tight")
            print(f"  [visualization] Guardado -> {salida}")
        plt.show()


# =============================================================================
# SECCIÓN 7 — SENSIBILIDAD OOS: CURVAS M2M MEDIAS POR VALOR DE PARÁMETRO
# =============================================================================

def plot_oos_parameter_sensitivity(
    wf_df: pd.DataFrame | None = None,
    param_keys: list[str] | None = None,
    save: bool = False,
    filename: str = "oos_parameter_sensitivity.png",
) -> None:
    """Curvas de equidad diaria OOS medias agrupadas por valor de hiperparámetro.

    DISEÑO METODOLÓGICO (v3)
    ─────────────────────────
    Fuente: walk_forward_comparison.csv — todos los ganadores evaluados OOS.

    Para cada hiperparámetro (ej. top_n, rebalance_freq, …):
      1. Identificar los valores únicos que toma dentro del pool de ganadores.
      2. Para cada valor (ej. top_n=1):
           a. Filtrar las configuraciones que tienen ese valor.
           b. Reconstruir su curva de rentabilidad OOS diaria (mark-to-market).
           c. Calcular la CURVA MEDIA diaria inter-estrategia del subgrupo.
      3. Dibujar una línea independiente por valor, normalizadas a 1.0 en el
         primer día hábil del período de test.

    De este modo se puede contrastar visualmente si, en entornos reales de
    mercado, la media de top_n=1 supera a top_n=3, etc.

    COHERENCIA DE ÍNDICES (v3)
    ──────────────────────────
    · elite_reset se construye UNA VEZ antes del bucle de reconstrucción.
    · Índice 0-based garantizado: curvas_por_idx[i] ↔ elite_reset.iloc[i].
    · Alineación temporal con bfill().ffill() para eliminar NaN líderes.

    No se incluye título superior ni textos flotantes interiores.

    Parámetros
    ----------
    wf_df      : tabla walk-forward (carga automática si es None).
    param_keys : hiperparámetros a analizar (auto-detectados si None).
    save       : guarda PNG en results/figures/ si es True.
    filename   : nombre del archivo de salida.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib no disponible.")
        return

    if wf_df is None:
        wf_df = load_walk_forward_results()
    if wf_df is None or wf_df.empty:
        print("[visualization] Sin datos walk-forward.")
        return

    # ── 1. Pool de ganadores con índice reseteado ─────────────────────────────
    # Usamos todos los ganadores de walk_forward_comparison.csv (no solo el
    # top 10% de ese ya-filtrado conjunto) para maximizar la base estadística.
    # Si el CSV tiene menos de 3 filas, se usa igualmente.
    elite_reset = wf_df.dropna(subset=["Sharpe_Test"]).copy().reset_index(drop=True)
    n_elite = len(elite_reset)
    print(f"[visualization] Pool OOS para sensibilidad: {n_elite} configuraciones")

    if n_elite == 0:
        print("[visualization] Sin configuraciones OOS disponibles.")
        return

    # ── 2. Columnas de parámetros ─────────────────────────────────────────────
    columnas_no_param = {
        "rank_insample", "rank_train", "config",
        "Sharpe_Train", "Sharpe_Test", "Degradacion_Sharpe", "Sharpe_Degradation_%",
        "CAGR_Train (%)", "CAGR_Test (%)", "CAGR_Train_%", "CAGR_Test_%",
        "MaxDD_Train (%)", "MaxDD_Test (%)", "MaxDD_Train_%", "MaxDD_Test_%",
    }
    if param_keys is None:
        param_keys = [c for c in elite_reset.columns if c not in columnas_no_param]

    if not param_keys:
        print("[visualization] No se encontraron columnas de hiperparámetros.")
        return

    price_wide = _get_price_wide()
    if price_wide is None or price_wide.empty:
        print("[visualization] Tabla de precios no disponible.")
        return

    test_inicio = pd.Timestamp(FECHA_TEST_START)
    test_fin    = price_wide.index.max().strftime("%Y-%m-%d")

    # ── 3. Reconstruir curvas OOS — una por configuración del pool ────────────
    # CRÍTICO: usamos enumerate() sobre las filas para garantizar que la posición
    # pos (0, 1, 2, ...) es SIEMPRE consistente, independientemente del índice
    # del DataFrame. Esto elimina cualquier riesgo de desajuste de índices.
    print("[visualization] Reconstruyendo curvas diarias OOS...")
    curvas_por_pos: dict[int, pd.Series] = {}
    failed_indices: list[int] = []

    for pos, fila in enumerate(elite_reset.itertuples(index=False)):
        pf_test = _run_engine_for_config(fila, (FECHA_TEST_START, test_fin))
        if pf_test.empty:
            failed_indices.append(pos)
            continue

        curva = reconstruct_daily_equity(pf_test, price_wide)
        if curva.empty:
            failed_indices.append(pos)
            continue

        fechas_test = curva.index[curva.index >= test_inicio]
        if len(fechas_test) == 0:
            failed_indices.append(pos)
            continue
        base = curva.loc[fechas_test[0]]
        if base == 0 or np.isnan(base):
            failed_indices.append(pos)
            continue

        curvas_por_pos[pos] = curva.loc[fechas_test[0]:] / base

    n_reconstruidas = len(curvas_por_pos)
    n_fallidas = len(failed_indices)
    print(f"  [visualization] {n_reconstruidas}/{n_elite} curvas reconstruidas ({n_fallidas} fallidas).")

    if n_reconstruidas == 0:
        print("[visualization] No se pudo reconstruir ninguna curva OOS diaria.")
        return

    # ── 4. Rejilla de fechas común + alineado bfill+ffill ────────────────────
    todas_fechas = pd.DatetimeIndex(
        sorted(set().union(*[set(s.index) for s in curvas_por_pos.values()]))
    )
    alineadas: dict[int, pd.Series] = {
        pos: s.reindex(todas_fechas).bfill().ffill()
        for pos, s in curvas_por_pos.items()
    }

    # ── 5. Validación de sincronización de índices ───────────────────────────
    # Verificar que todas las claves en `alineadas` son posiciones válidas
    # dentro del rango de elite_reset (0 a n_elite-1).
    posiciones_esperadas = set(range(n_elite))
    posiciones_reales = set(alineadas.keys())
    posiciones_invalidas = posiciones_reales - posiciones_esperadas
    if posiciones_invalidas:
        print(f"  [AVISO CRÍTICO] {len(posiciones_invalidas)} índices fuera de rango detectados.")
        print(f"  Rango esperado: 0-{n_elite-1}, índices inválidos: {sorted(posiciones_invalidas)[:10]}...")

    # ── 6. Pre-computar conteos totales por parámetro para labels precisos ────
    # Para cada parámetro, pre-calculamos cuántas estrategias tienen cada valor
    # (del pool completo), para que el label muestre "n=reconstruidas/total".
    conteos_totales_por_param: dict[str, dict] = {}
    for param in param_keys:
        if param not in elite_reset.columns:
            continue
        conteos_totales_por_param[param] = elite_reset[param].value_counts().to_dict()

    # ── 5. Gráficas por hiperparámetro ────────────────────────────────────────
    ncols = min(3, len(param_keys))
    nrows = (len(param_keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows), squeeze=False)
    ejes_flat = axes.flatten()

    paleta = [
        "#1f4e79", "#e07b39", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for idx_ax, param in enumerate(param_keys):
        ax = ejes_flat[idx_ax]

        if param not in elite_reset.columns:
            ax.set_visible(False)
            continue

        valores_unicos = sorted(
            elite_reset[param].dropna().unique(),
            key=lambda v: (str(type(v).__name__), v),
        )

        curvas_dibujadas = 0
        for idx_col, valor in enumerate(valores_unicos):
            # Máscara sobre elite_reset usando posición posicional (iloc)
            # para evitar cualquier desajuste de índice.
            mask_values = elite_reset[param].values == valor
            posiciones_grupo = [int(p) for p, m in enumerate(mask_values) if m]

            curvas_g = [alineadas[p] for p in posiciones_grupo if p in alineadas]
            n_total_grupo = len(posiciones_grupo)
            n_reconstruidas_grupo = len(curvas_g)

            if not curvas_g:
                continue

            df_grupo    = pd.DataFrame(dict(enumerate(curvas_g)))
            curva_media = df_grupo.mean(axis=1, skipna=True).dropna()

            if curva_media.empty:
                continue

            color    = paleta[idx_col % len(paleta)]
            ax.plot(curva_media.index, curva_media.values,
                    color=color, linewidth=2.0,
                    label=f"{valor}  (n={n_reconstruidas_grupo}/{n_total_grupo})", zorder=5)
            curvas_dibujadas += 1

        if curvas_dibujadas == 0:
            ax.set_visible(False)
            continue

        ax.axvline(test_inicio, color="black", linestyle=":", linewidth=1.5, zorder=6)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=3)

        ax.set_ylabel("Retorno Acumulado (base = 1.0)", fontsize=9)
        ax.set_xlabel(param, fontsize=9)
        ax.legend(fontsize=8, loc="upper left", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(param_keys), len(ejes_flat)):
        ejes_flat[j].set_visible(False)

    plt.tight_layout()

    if save:
        ensure_dirs()
        salida = os.path.join(DIR_FIGURES, filename)
        plt.savefig(salida, dpi=150, bbox_inches="tight")
        print(f"  [visualization] Guardado -> {salida}")
    plt.show()
