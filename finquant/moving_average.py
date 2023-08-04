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

from finquant.data_types import SERIES_OR_DATAFRAME
from finquant.type_utilities import type_validation


def compute_ma(
    data: SERIES_OR_DATAFRAME,
    fun: Callable[[SERIES_OR_DATAFRAME, int], pd.Series],
    spans: List[int],
    plot: bool = True,
) -> pd.DataFrame:
    """Computes a band of moving averages (sma or ema, depends on the input argument
    `fun`) for a number of different time windows. If `plot` is `True`, it also
    computes and sets markers for buy/sell signals based on crossovers of the Moving
    Averages with the shortest/longest spans.

    :param data: A series/dataframe of daily stock prices (if ``pandas.DataFrame``,
        only one column is expected)
    :type data: :py:data:`~.finquant.type_definitions.SERIES_OR_DATAFRAME`
    :param fun: Function that computes a moving average, e.g. ```sma``` (simple) or
         ```ema``` (exponential).
    :param spans: List of integers, time windows to compute the Moving Average on.
    :param plot: boolean, whether to plot the moving averages
         and buy/sell signals based on crossovers of shortest and longest
         moving average, or not, default: True

    :return: Moving averages of given data.
    """
    # Type validations:
    type_validation(data=data, fun=fun, spans=spans, plot=plot)
    m_a = data.copy(deep=True)
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(m_a, pd.Series):
        m_a = m_a.to_frame()
    # compute moving averages
    for span in spans:
        m_a[str(span) + "d"] = fun(data, span)
    if plot:
        fig = plt.figure()
        axis = fig.add_subplot(111)
        # plot moving averages
        m_a.plot(ax=axis)
        # Create buy/sell signals of shortest and longest span
        minspan = min(spans)
        minlabel = str(minspan) + "d"
        maxspan = max(spans)
        maxlabel = str(maxspan) + "d"
        signals = m_a.copy(deep=True)
        signals["diff"] = 0.0
        signals["diff"][minspan:] = np.where(
            m_a[minlabel][minspan:] > m_a[maxlabel][minspan:], 1.0, 0.0
        )
        # Generate trading orders
        signals["signal"] = signals["diff"].diff()
        # marker for buy signal
        axis.plot(
            signals.loc[signals["signal"] == 1.0].index.values,
            signals[minlabel][signals["signal"] == 1.0].values,
            marker="^",
            markersize=10,
            color="r",
            label="buy signal",
        )
        # marker for sell signal
        axis.plot(
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
    return m_a


def sma(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the simple moving average.

    Note: the moving average is computed on all columns.

    :param data: A dataframe of daily stock prices
    :param span: Number of days/values over which the average is computed, default: 100

    :return: ``sma``: simple moving average
    """
    # Type validations:
    type_validation(data=data, span=span)
    return data.rolling(window=span, center=False).mean()


def ema(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the exponential moving average.

    Note: the moving average is computed on all columns.

    :param data: A dataframe of daily stock prices
    :param span: Number of days/values over which the average is computed, default: 100

    :return: ``ema``: exponential moving average
    """
    # Type validations:
    type_validation(data=data, span=span)
    return data.ewm(span=span, adjust=False, min_periods=span).mean()


def sma_std(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the standard deviation of the simple moving
    average.

    :param data: A dataframe of daily stock prices
    :param span: Number of days/values over which the average is computed, default: 100

    :return: ``sma_std``: standard deviation of simple moving average
    """
    # Type validations:
    type_validation(data=data, span=span)
    return data.rolling(window=span, center=False).std()


def ema_std(data: pd.DataFrame, span: int = 100) -> pd.DataFrame:
    """Computes and returns the standard deviation of the exponential
    moving average.

    :param data: A dataframe of daily stock prices
    :param span: Number of days/values over which the average is computed, default: 100

    :return: ``ema_std``: standard deviation of exponential moving average
    """
    # Type validations:
    type_validation(data=data, span=span)
    return data.ewm(span=span, adjust=False, min_periods=span).std()


def plot_bollinger_band(
    data: pd.DataFrame,
    fun: Callable[[pd.DataFrame, int], pd.DataFrame],
    span: int = 100,
) -> None:
    """Computes and visualises a Bolling Band.

    :param data: A dataframe of daily stock prices
    :param fun: function that computes a moving average, e.g. ``sma`` (simple) or
         ``ema`` (exponential).
    :param span: Number of days/values over which the average is computed, default: 100
    """
    # Type validations:
    type_validation(data=data, fun=fun, span=span)
    # special requirement for dataframe "data":
    if isinstance(data, pd.DataFrame) and not len(data.columns.values) == 1:
        raise ValueError("data is expected to have only one column.")
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # compute moving average
    m_a = compute_ma(data, fun, [span], plot=False)
    # create dataframes for bollinger band object and standard
    # deviation
    bol = m_a.copy(deep=True)
    std = m_a.copy(deep=True)
    # get column label
    collabel = data.columns.values[0]
    # get standard deviation
    if fun is sma:
        std[str(span) + "d std"] = sma_std(data[collabel], span=span)
    elif fun is ema:
        std[str(span) + "d std"] = ema_std(data[collabel], span=span)
    # compute upper and lower band
    bol["Lower Band"] = bol[str(span) + "d"] - (std[str(span) + "d std"] * 2)
    bol["Upper Band"] = bol[str(span) + "d"] + (std[str(span) + "d std"] * 2)
    # plot
    fig = plt.figure()
    axis = fig.add_subplot(111)
    # bollinger band
    axis.fill_between(
        data.index.values,
        bol["Upper Band"],
        bol["Lower Band"],
        color="darkgrey",
        label="Bollinger Band",
    )
    # plot data and moving average
    bol[collabel].plot(ax=axis)
    bol[str(span) + "d"].plot(ax=axis)
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
