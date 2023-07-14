"""The module provides functions to compute and visualise:

- *Simple Moving Averages*,
- *Exponential Moving Averages*,
- a *band* of *Moving Averages* (simple or exponential), and
- *Bollinger Bands*.
"""


from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_ma(
    data, fun: Callable, spans: List[int], plot: bool = True
) -> pd.DataFrame:
    """Computes a band of moving averages (sma or ema, depends on the input argument
    `fun`) for a number of different time windows. If `plot` is `True`, it also
    computes and sets markers for buy/sell signals based on crossovers of the Moving
    Averages with the shortest/longest spans.

    :Input:
     :data: pandas.DataFrame, or pandas.Series, with stock prices
         (if pandas.DataFrame: only one column is expected)
     :fun: function that computes a moving average, e.g. sma (simple) or
         ema (exponential).
     :spans: list of integers, time windows to compute the Moving Average on.
     :plot: boolean (default: True), whether to plot the moving averages
         and buy/sell signals based on crossovers of shortest and longest
         moving average.

    :Output:
     :ma: pandas.DataFrame with moving averages of given data.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "data is expected to be of type pandas.Series or pandas.DataFrame"
        )
    ma = data.copy(deep=True)
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(ma, pd.Series):
        ma = ma.to_frame()
    # compute moving averages
    for span in spans:
        ma[str(span) + "d"] = fun(data, span=span)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plot moving averages
        ma.plot(ax=ax)
        # Create buy/sell signals of shortest and longest span
        minspan = min(spans)
        minlabel = str(minspan) + "d"
        maxspan = max(spans)
        maxlabel = str(maxspan) + "d"
        signals = ma.copy(deep=True)
        signals["diff"] = 0.0
        signals["diff"][minspan:] = np.where(
            ma[minlabel][minspan:] > ma[maxlabel][minspan:], 1.0, 0.0
        )
        # Generate trading orders
        signals["signal"] = signals["diff"].diff()
        # marker for buy signal
        ax.plot(
            signals.loc[signals["signal"] == 1.0].index.values,
            signals[minlabel][signals["signal"] == 1.0].values,
            marker="^",
            markersize=10,
            color="r",
            label="buy signal",
        )
        # marker for sell signal
        ax.plot(
            signals.loc[signals["signal"] == -1.0].index.values,
            signals[minlabel][signals["signal"] == -1.0].values,
            marker="v",
            markersize=10,
            color="b",
            label="sell signal",
        )
        # title
        title = "Band of Moving Averages (" + str(fun.__name__) + ")"
        plt.title(title)
        # legend
        plt.legend(ncol=2)
        # axis labels
        plt.xlabel(data.index.name)
        plt.ylabel("Price")
    return ma


def sma(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the simple moving average.

    Note: the moving average is computed on all columns.

    :Input:
     :data: pandas.DataFrame with stock prices in columns
     :span: int (default: 100), number of days/values over which
         the average is computed

    :Output:
     :sma: pandas.DataFrame of simple moving average
    """
    return data.rolling(window=span, center=False).mean()


def ema(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the exponential moving average.

    Note: the moving average is computed on all columns.

    :Input:
     :data: pandas.DataFrame with stock prices in columns
     :span: int (default: 100), number of days/values over which
         the average is computed

    :Output:
     :ema: pandas.DataFrame of exponential moving average
    """
    return data.ewm(span=span, adjust=False, min_periods=span).mean()


def sma_std(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the standard deviation of the simple moving
    average.

    :Input:
     :data: pandas.DataFrame with stock prices in columns
     :span: int (default: 100), number of days/values over which
         the average is computed

    :Output:
     :sma_std: pandas.DataFrame of standard deviation of
         simple moving average
    """
    return data.rolling(window=span, center=False).std()


def ema_std(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the standard deviation of the exponential
    moving average.

    :Input:
     :data: pandas.DataFrame with stock prices in columns
     :span: int (default: 100), number of days/values over which
         the average is computed

    :Output:
     :ema_std: pandas.DataFrame of standard deviation of
         exponential moving average
    """
    return data.ewm(span=span, adjust=False, min_periods=span).std()


def plot_bollinger_band(data, fun: Callable, span: int = 100) -> None:
    """Computes and visualises a Bolling Band.

    :Input:
     :data: pandas.Series or pandas.DataFrame with stock prices in columns
     :fun: function that computes a moving average, e.g. sma (simple) or
         ema (exponential).
     :span: int (default: 100), number of days/values over which
         the average is computed
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "data is expected to be of type pandas.Series or pandas.DataFrame"
        )
    if isinstance(data, pd.DataFrame) and not len(data.columns.values) == 1:
        raise ValueError("data is expected to have only one column.")
    if not isinstance(span, int):
        raise ValueError("span must be an integer.")
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # compute moving average
    ma = compute_ma(data, fun, [span], plot=False)
    # create dataframes for bollinger band object and standard
    # deviation
    bol = ma.copy(deep=True)
    std = ma.copy(deep=True)
    # get column label
    collabel = data.columns.values[0]
    # get standard deviation
    if fun == sma:
        std[str(span) + "d std"] = sma_std(data[collabel], span=span)
    elif fun == ema:
        std[str(span) + "d std"] = ema_std(data[collabel], span=span)
    # compute upper and lower band
    bol["Lower Band"] = bol[str(span) + "d"] - (std[str(span) + "d std"] * 2)
    bol["Upper Band"] = bol[str(span) + "d"] + (std[str(span) + "d std"] * 2)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # bollinger band
    ax.fill_between(
        data.index.values,
        bol["Upper Band"],
        bol["Lower Band"],
        color="darkgrey",
        label="Bollinger Band",
    )
    # plot data and moving average
    bol[collabel].plot(ax=ax)
    bol[str(span) + "d"].plot(ax=ax)
    # title
    title = (
        "Bollinger Band of +/- 2$\\sigma$, Moving Average of "
        + str(fun.__name__)
        + " over "
        + str(span)
        + " days"
    )
    plt.title(title)
    # legend
    plt.legend()
    # axis labels
    plt.xlabel(data.index.name)
    plt.ylabel("Price")
