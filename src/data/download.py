import yfinance as yf
import pandas as pd


def descargar_datos(tickers, start_date):
    """
    Descarga datos OHLCV ajustados desde Yahoo Finance.
    """
    raw = yf.download(
        tickers,
        start=start_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    return {
        "close": raw["Close"],
        "high": raw["High"],
        "low": raw["Low"],
        "volume": raw["Volume"],
    }