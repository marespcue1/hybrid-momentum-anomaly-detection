# src/features/indicators.py

import pandas as pd
import ta


def add_rsi(df, window=14):
    df = df.copy()
    df["RSI"] = df.groupby("Ticker")["Precio_Adj"].transform(
        lambda x: ta.momentum.RSIIndicator(x, window=window).rsi()
    )
    return df


def add_macd(df, window_slow=26, window_fast=12, window_sign=9):
    df = df.copy()

    def _macd(x):
        macd = ta.trend.MACD(
            x,
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        return pd.DataFrame({
            "MACD": macd.macd(),
            "MACD_signal": macd.macd_signal()
        })

    macd_df = df.groupby("Ticker")["Precio_Adj"].apply(_macd).reset_index(level=0, drop=True)

    df[["MACD", "MACD_signal"]] = macd_df

    return df


def add_volatility(df, window=21):
    df = df.copy()
    df["Volatility"] = df.groupby("Ticker")["Return"].rolling(window).std().reset_index(0, drop=True)
    return df