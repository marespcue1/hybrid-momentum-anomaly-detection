def compute_returns(df):
    df["Return"] = df.groupby("Ticker")["Precio_Adj"].pct_change()
    return df


def compute_volatility(df, window=21):
    df["Volatility"] = (
        df.groupby("Ticker")["Return"]
        .rolling(window)
        .std()
        .reset_index(0, drop=True)
    )
    return df