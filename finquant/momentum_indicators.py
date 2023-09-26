""" This module provides function(s) to compute momentum indicators
used in technical analysis such as RSI, MACD etc. """

import matplotlib.pyplot as plt
import pandas as pd


def relative_strength_index(
    data,
    window_length: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    standalone: bool = False,
) -> None:
    """Computes and visualizes a RSI graph,
        plotted along with the prices in another sub-graph
        for comparison.

     Ref: https://www.investopedia.com/terms/r/rsi.asp

    :Input
     :data: pandas.Series or pandas.DataFrame with stock prices in columns
     :window_length: Window length to compute RSI, default being 14 days
     :oversold: Standard level for oversold RSI, default being 30
     :overbought: Standard level for overbought RSI, default being 70
     :standalone: Plot only the RSI graph
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "data is expected to be of type pandas.Series or pandas.DataFrame"
        )
    if isinstance(data, pd.DataFrame) and not len(data.columns.values) == 1:
        raise ValueError("data is expected to have only one column.")
    # checking integer fields
    for field in (window_length, oversold, overbought):
        if not isinstance(field, int):
            raise ValueError(f"{field} must be an integer.")
    # validating levels
    if oversold >= overbought:
        raise ValueError("oversold level should be < overbought level")
    if not (0 < oversold < 100) or not(0 < overbought < 100):
        raise ValueError("levels should be > 0 and < 100")
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # get the stock key
    stock = data.keys()[0]
    # calculate price differences
    data["diff"] = data.diff(1)
    # calculate gains and losses
    data["gain"] = data["diff"].clip(lower=0).round(2)
    data["loss"] = data["diff"].clip(upper=0).abs().round(2)
    # placeholder
    wl = window_length
    # calculate rolling window mean gains and losses
    data["avg_gain"] = data["gain"].rolling(window=wl, min_periods=wl).mean()
    data["avg_loss"] = data["loss"].rolling(window=wl, min_periods=wl).mean()
    # calculate WMS (wilder smoothing method) averages
    for i, row in enumerate(data["avg_gain"].iloc[wl + 1 :]):
        data["avg_gain"].iloc[i + wl + 1] = (
            data["avg_gain"].iloc[i + wl] * (wl - 1) + data["gain"].iloc[i + wl + 1]
        ) / wl
    for i, row in enumerate(data["avg_loss"].iloc[wl + 1 :]):
        data["avg_loss"].iloc[i + wl + 1] = (
            data["avg_loss"].iloc[i + wl] * (wl - 1) + data["loss"].iloc[i + wl + 1]
        ) / wl
    # calculate RS values
    data["rs"] = data["avg_gain"] / data["avg_loss"]
    # calculate RSI
    data["rsi"] = 100 - (100 / (1.0 + data["rs"]))
    # Plot it
    if standalone:
        # Single plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axhline(y=oversold, color="g", linestyle="--")
        ax.axhline(y=overbought, color="r", linestyle="--")
        data["rsi"].plot(ylabel="RSI", xlabel="Date", ax=ax, grid=True)
        plt.title("RSI Plot")
        plt.legend()
    else:
        # RSI against price in 2 plots
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
        ax[0].axhline(y=oversold, color="g", linestyle="--")
        ax[0].axhline(y=overbought, color="r", linestyle="--")
        ax[0].set_title("RSI + Price Plot")
        # plot 2 graphs in 2 colors
        colors = plt.rcParams["axes.prop_cycle"]()
        data["rsi"].plot(
            ylabel="RSI", ax=ax[0], grid=True, color=next(colors)["color"], legend=True
        )
        data[stock].plot(
            xlabel="Date",
            ylabel="Price",
            ax=ax[1],
            grid=True,
            color=next(colors)["color"],
            legend=True,
        )
        plt.legend()


def macd(
    data,
    longer_ema_window: int = 26,
    shorter_ema_window: int = 12,
    signal_ema_window: int = 9,
    standalone: bool = False,
) -> None:
    """
    Computes and visualizes a MACD (Moving Average Convergence Divergence)
    plotted along with price chart in another sub-graph for comparison.

    Ref: https://www.alpharithms.com/calculate-macd-python-272222/

    :Input
     :data: pandas.Series or pandas.DataFrame with stock prices in columns
     :longer_ema_window:  Window length (in days) for the longer EMA
     :shorter_ema_window: Window length (in days) for the shorter EMA
     :signal_ema_window:  Window length (in days) for the signal
     :standalone: If true, plot only the MACD signal
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "data is expected to be of type pandas.Series or pandas.DataFrame"
        )
    if isinstance(data, pd.DataFrame) and not len(data.columns.values) == 1:
        raise ValueError("data is expected to have only one column.")
    # checking integer fields
    for field in (longer_ema_window, shorter_ema_window, signal_ema_window):
        if not isinstance(field, int):
            raise ValueError(f"{field} must be an integer.")
    # validating windows
    if longer_ema_window < shorter_ema_window:
        raise ValueError("longer ema window should be > shorter ema window")
    if longer_ema_window < signal_ema_window:
        raise ValueError("longer ema window should be > signal ema window")

    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # get the stock key
    stock = data.keys()[0]
    # calculate EMA short period
    ema_short = data.ewm(
        span=shorter_ema_window, adjust=False, min_periods=shorter_ema_window
    ).mean()
    # calculate EMA long period
    ema_long = data.ewm(
        span=longer_ema_window, adjust=False, min_periods=longer_ema_window
    ).mean()
    # Subtract the longwer window EMA from the shorter window EMA to get the MACD
    data["macd"] = ema_long - ema_short
    # Get the signal window MACD for the Trigger line
    data["macd_s"] = (
        data["macd"]
        .ewm(span=signal_ema_window, adjust=False, min_periods=signal_ema_window)
        .mean()
    )
    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    data["diff"] = data["macd"] - data["macd_s"]
    hist = data["diff"]

    # Plot it
    if standalone:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data["macd"].plot(
            ylabel="MACD",
            xlabel="Date",
            ax=ax,
            grid=True,
            label="MACD",
            color="green",
            linewidth=1.5,
            legend=True,
        )
        hist.plot(
            ax=ax, grid=True, label="diff", color="black", linewidth=0.5, legend=True
        )
        data["macd_s"].plot(
            ax=ax, grid=True, label="SIGNAL", color="red", linewidth=1.5, legend=True
        )

        for i, key in enumerate(hist.index):
            if hist[key] < 0:
                ax.bar(data.index[i], hist[key], color="orange")
            else:
                ax.bar(data.index[i], hist[key], color="black")
    else:
        # RSI against price in 2 plots
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
        ax[0].set_title("MACD + Price Plot")
        data["macd"].plot(
            ylabel="MACD",
            xlabel="Date",
            ax=ax[0],
            grid=True,
            label="MACD",
            color="green",
            linewidth=1.5,
            legend=True,
        )
        hist.plot(
            ax=ax[0], grid=True, label="diff", color="black", linewidth=0.5, legend=True
        )
        data["macd_s"].plot(
            ax=ax[0], grid=True, label="SIGNAL", color="red", linewidth=1.5, legend=True
        )

        for i, key in enumerate(hist.index):
            if hist[key] < 0:
                ax.bar(data.index[i], hist[key], color="orange")
            else:
                ax.bar(data.index[i], hist[key], color="black")

        data[stock].plot(
            xlabel="Date",
            ylabel="Price",
            ax=ax[1],
            grid=True,
            color="orange",
            legend=True,
        )
        plt.legend()
