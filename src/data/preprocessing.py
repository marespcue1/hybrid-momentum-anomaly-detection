# src/data/preprocessing.py

import pandas as pd


def unificar_series_temporales(close_df, high_df, low_df, volume_df):
    df = pd.concat({
        "Precio_Adj": close_df,
        "High": high_df,
        "Low": low_df,
        "Volumen": volume_df
    }, axis=1)

    df = df.stack(level=1).reset_index()
    df.columns = ["Fecha", "Ticker", "Precio_Adj", "High", "Low", "Volumen"]

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(["Ticker", "Fecha"]).reset_index(drop=True)

    return df


def compute_returns(df):
    df = df.copy()
    df["Return"] = df.groupby("Ticker")["Precio_Adj"].pct_change()
    return df


def drop_missing_basic(df):
    """
    Limpieza mínima inicial (sin cargarte información útil).
    """
    return df.dropna(subset=["Precio_Adj"])


def filter_min_observations(df, min_obs=200):
    """
    Elimina activos con poco histórico.
    """
    counts = df.groupby("Ticker").size()
    valid = counts[counts >= min_obs].index
    return df[df["Ticker"].isin(valid)]