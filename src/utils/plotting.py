import matplotlib.pyplot as plt
import seaborn as sns

def plot_normalized_prices(df, ticker_names=None, top_n=5):

    df = df.copy()

    # Normalización
    df["Precio_Norm"] = df.groupby("Ticker")["Precio_Adj"].transform(
        lambda x: x / x.iloc[0]
    )

    ticker_rank = (
        df.groupby("Ticker")["Precio_Norm"]
        .sum()
        .sort_values(ascending=False)
    )

    top_tickers = ticker_rank.head(top_n).index

    plt.figure(figsize=(12, 6))

    for ticker in df["Ticker"].unique():

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

    df["Precio_Norm"] = df.groupby("Ticker")["Precio_Adj"].transform(
        lambda x: x / x.iloc[0]
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    groups = ["spain", "usa", "etfs"]

    for i, group in enumerate(groups):

        for j, ticker in enumerate(tickers_dict[group]):

            subset = df[df["Ticker"] == ticker]
            label_name = ticker_names.get(ticker, ticker)
            if j < 10:
                axes[i].plot(
                    subset["Fecha"],
                    subset["Precio_Norm"],
                    alpha=0.8,
                    label=f"{label_name} ({ticker})"
                )
            else:
                axes[i].plot(
                    subset["Fecha"],
                    subset["Precio_Norm"],
                    alpha=0.4
                )

        axes[i].set_title(group.upper())
        axes[i].set_ylabel("Precio normalizado")
        axes[i].legend()

    axes[-1].set_xlabel("Fecha")

    plt.tight_layout()
    plt.show()


def plot_returns_distribution(df):

    plt.figure(figsize=(10,5))
    sns.histplot(df["Return"].dropna(), bins=100, kde=True)
    plt.title("Distribución de retornos")
    plt.show()


def plot_correlation_matrix(df, ticker_names=None):

    pivot = df.pivot(index="Fecha", columns="Ticker", values="Return")
    corr = pivot.corr()

    # Renombrar columnas si hay diccionario
    if ticker_names:
        corr.columns = [f"{ticker_names.get(c, c)} ({c})" for c in corr.columns]
        corr.index = corr.columns

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