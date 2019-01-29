"""The module provides functions to compute different kinds of returns of stocks."""


import numpy as np
import pandas as pd


def cumulative_returns(data, dividend=0):
    """Returns DataFrame with cumulative returns

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}{\\text{price}_{t_0}}`

    :Input:
     :data: pandas.DataFrame with daily stock prices
     :dividend: float (default: 0), paid dividend

    :Output:
     :returns: a pandas.DataFrame of cumulative returns of given stock prices
    """
    return data.apply(lambda x: (x - x[0] + dividend) / x[0])


def daily_returns(data):
    """Returns DataFrame with daily returns (percentage change)

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}`

    :Input:
     :data: pandas.DataFrame with daily stock prices

    :Output:
     :returns: a pandas.DataFrame of daily percentage change of returns
         of given stock prices
    """
    return data.pct_change().dropna(how="all")


def daily_log_returns(data):
    """
    Returns DataFrame with daily log returns

    :math:`R_{\\log} = \\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}\\right)`

    :Input:
     :data: pandas.DataFrame with daily stock prices

    :Output:
     :returns: a pandas.DataFrame of
         log(1 + daily percentage change of returns)
    """
    return np.log(1 + daily_returns(data)).dropna(how="all")


def historical_mean_return(data, freq=252):
    """Returns the mean return based on historical stock price data.

    :Input:
     :data: pandas.DataFrame with daily stock prices
     :freq: int (default: 252), number of trading days, default
             value corresponds to trading days in a year

    :Output:
     :returns: a pandas.DataFrame of mean daily * freq
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas.DataFrame")
    return daily_returns(data).mean() * freq
