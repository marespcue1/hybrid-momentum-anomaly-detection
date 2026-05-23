# =============================================================================
# src/analysis/parameter_analysis.py
# =============================================================================

from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_walkforward_results(path):

    df = pd.read_csv(
        path,
        sep=";",
        decimal=","
    )

    return df


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

def parameter_summary(
    df,
    parameter,
):

    summary = (
        df.groupby(parameter)["Sharpe_Test"]
        .agg([
            "mean",
            "median",
            "std",
            "max",
            "count",
        ])
        .sort_values("mean", ascending=False)
    )

    return summary


# =============================================================================
# TOP CONFIGURATIONS
# =============================================================================

def get_top_configurations(
    df,
    n=20,
):

    return (
        df.sort_values(
            "Sharpe_Test",
            ascending=False,
        )
        .head(n)
    )


# =============================================================================
# ROBUST CONFIGURATIONS
# =============================================================================

def get_robust_configurations(
    df,
    min_sharpe=0.5,
    max_degradation=0.5,
):

    robust = df[
        (df["Sharpe_Test"] >= min_sharpe)
        &
        (df["Degradacion_Sharpe"] <= max_degradation)
    ]

    robust = robust.sort_values(
        "Sharpe_Test",
        ascending=False,
    )

    return robust


# =============================================================================
# OVERFITTING SCORE
# =============================================================================

def compute_overfitting_score(df):

    df = df.copy()

    df["Overfit_Score"] = (
        df["Sharpe_Train"]
        -
        df["Sharpe_Test"]
    )

    return df


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(
    df,
    params,
):

    ml_df = df.copy()

    # =========================================================
    # Nos quedamos solo con columnas necesarias
    # =========================================================

    cols_needed = params + ["Sharpe_Test"]

    ml_df = ml_df[cols_needed].copy()

    # =========================================================
    # Eliminar filas con NaNs
    # =========================================================

    ml_df = ml_df.dropna()

    # =========================================================
    # Convertir categóricas a números
    # =========================================================

    encoders = {}

    for col in params:

        if not pd.api.types.is_numeric_dtype(ml_df[col]):

            le = LabelEncoder()

            ml_df[col] = le.fit_transform(
                ml_df[col].astype(str)
            )

            encoders[col] = le

    # =========================================================
    # Features / Target
    # =========================================================

    X = ml_df[params]

    y = ml_df["Sharpe_Test"]

    # =========================================================
    # Modelo
    # =========================================================

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    # =========================================================
    # Importancias
    # =========================================================

    importance_df = pd.DataFrame({
        "Variable": X.columns,
        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    )

    return importance_df


# =============================================================================
# TOP VALUE FREQUENCIES
# =============================================================================

def top_value_frequencies(
    df,
    params,
    top_n=50,
):

    top_df = get_top_configurations(
        df,
        n=top_n,
    )

    results = {}

    for p in params:

        freq = (
            top_df[p]
            .value_counts(normalize=True)
            .sort_values(ascending=False)
        )

        results[p] = freq

    return results