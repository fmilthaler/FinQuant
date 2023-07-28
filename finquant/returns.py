"""The module provides functions to compute different kinds of returns of stocks."""


import numpy as np
import pandas as pd

from finquant.type_definitions import (
    INT,
    NUMERIC,
)


def cumulative_returns(data: pd.DataFrame, dividend: NUMERIC = 0) -> pd.DataFrame:
    """Returns DataFrame with cumulative returns

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}
    {\\text{price}_{t_0}}`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices
     :dividend: ``float`` (default= ``0``), paid dividend

    :Output:
     :ret: a ``pandas.DataFrame`` of cumulative Returns of given stock prices.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return data.dropna(axis=0, how="any").apply(lambda x: (x - x[0] + dividend) / x[0])


def daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with daily returns (percentage change)

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices

    :Output:
     :ret: a ``pandas.DataFrame`` of daily percentage change of Returns
         of given stock prices.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return data.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan)


def daily_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with daily log returns

    :math:`R_{\\log} = \\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}
    {\\text{price}_{t_{i-1}}}\\right)`

    :Input:
     :data: ``pandas.DataFrame`` with daily stock prices

    :Output:
     :ret: a ``pandas.DataFrame`` of
         log(1 + daily percentage change of Returns)
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return np.log(1 + daily_returns(data)).dropna(how="all")


def historical_mean_return(data: pd.DataFrame, freq: INT = 252) -> pd.Series:
    """Returns the mean return based on historical stock price data.

    :Input:
     :data: ``pandas.DataFrame`` or ``pandas.Series`` with daily stock prices
     :freq: ``int`` (default= ``252``), number of trading days, default
             value corresponds to trading days in a year

    :Output:
     :ret: a ``pandas.Series`` or ``numpy.float`` of historical mean Returns.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return daily_returns(data).mean() * freq
