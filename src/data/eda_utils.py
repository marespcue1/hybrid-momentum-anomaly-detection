# src/data/eda_utils.py

import pandas as pd


def missing_summary(df):
    return df.isna().sum()


def missing_by_ticker(df):
    return df.groupby("Ticker").apply(lambda x: x.isna().mean(numeric_only=True))


def observations_per_ticker(df):
    return df.groupby("Ticker").size().sort_values(ascending=False)


def returns_stats(df):
    return df.groupby("Ticker")["Return"].agg(["mean", "std", "skew", "kurt"])


def correlation_matrix(df):
    pivot = df.pivot(index="Fecha", columns="Ticker", values="Return")
    return pivot.corr()