import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np

def plot_normalized_prices(df, ticker_names=None, top_n=5):

    df = df.copy()

    df["Precio_Norm"] = df.groupby("Ticker")["Precio_Adj"].transform(
        lambda x: x / x.iloc[0]
    )

    ticker_rank = (
        df.groupby("Ticker")["Precio_Norm"]
        .last()
        .sort_values(ascending=False)
    )

    top_tickers = ticker_rank.head(top_n).index

    ordered_tickers = ticker_rank.index

    plt.figure(figsize=(12, 6))

    for ticker in ordered_tickers:

        subset = df[df["Ticker"] == ticker]

        label = ticker_names.get(ticker, ticker) if ticker_names else ticker

        if ticker in top_tickers:
            plt.plot(
                subset["Fecha"],
                subset["Precio_Norm"],
                linewidth=2,
                label=f"{label} ({ticker})"
            )
        else:
            plt.plot(
                subset["Fecha"],
                subset["Precio_Norm"],
                alpha=0.2,
                color="gray"
            )

    plt.title(f"Precios normalizados (Top {top_n} por área)")
    plt.legend()
    plt.show()

def plot_normalized_prices_by_group(df, tickers_dict, ticker_names):

    df = df.copy()

    df["Precio_Norm"] = (
        df.groupby("Ticker")["Precio_Adj"]
        .transform(lambda x: x / x.iloc[0])
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True
    )

    groups = ["spain", "usa", "etfs"]

    for i, group in enumerate(groups):

        group_tickers = tickers_dict[group]

        final_values = (
            df[df["Ticker"].isin(group_tickers)]
            .groupby("Ticker")["Precio_Norm"]
            .last()
            .sort_values(ascending=False)
        )

        ordered_tickers = final_values.index

        for ticker in ordered_tickers:

            subset = df[df["Ticker"] == ticker]

            label = ticker_names.get(ticker, ticker)

            axes[i].plot(
                subset["Fecha"],
                subset["Precio_Norm"],
                label=f"{label} ({ticker})"
            )

        axes[i].set_title(group.upper())
        axes[i].set_ylabel("Precio normalizado")
        axes[i].legend(
            loc="upper left",
            fontsize=8
        )

    axes[-1].set_xlabel("Fecha")

    plt.tight_layout()
    plt.show()


def plot_returns_distribution(df):

    plt.figure(figsize=(10,5))
    sns.histplot(df["Return"].dropna(), bins=100, kde=True)
    plt.title("Distribución de retornos")
    plt.show()

def plot_QQ_probplot(df):
    stats.probplot(
        df["Return"].dropna(),
        dist="norm",
        plot=plt
    )

    plt.title("Gráfico Q-Q de los retornos")
    plt.xlabel("Cuantiles teóricos de la distribución normal")
    plt.ylabel("Cuantiles observados")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_correlation_matrix(df, ticker_names=None):

    pivot = df.pivot(
        index="Fecha",
        columns="Ticker",
        values="Return"
    )

    corr = pivot.corr()

    distance_matrix = 1 - np.abs(corr)

    condensed_distance = squareform(
        distance_matrix,
        checks=False
    )

    linkage_matrix = linkage(
        condensed_distance,
        method="average"
    )

    order = leaves_list(linkage_matrix)

    corr = corr.iloc[order, order]

    if ticker_names:
        labels = [
            f"{ticker_names.get(c, c)} ({c})"
            for c in corr.columns
        ]
        corr.columns = labels
        corr.index = labels

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)

    plt.title("Matriz de correlación de retornos")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr


def plot_obs_per_ticker(df):

    obs = df.groupby("Ticker").size().sort_values()

    plt.figure()
    obs.plot(kind="bar")
    plt.title("Número de observaciones por activo")
    plt.xticks(rotation=90)
    plt.show()