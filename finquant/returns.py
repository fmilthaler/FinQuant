"""The module provides functions to compute different kinds of returns of stocks."""


from typing import Union

import numpy as np
import pandas as pd

from finquant.type_definitions import INT, NUMERIC, SERIES_OR_DATAFRAME


def cumulative_returns(data: pd.DataFrame, dividend: NUMERIC = 0) -> pd.DataFrame:
    """Returns DataFrame with cumulative returns

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}
    {\\text{price}_{t_0}}`

    :param data: A dataframe of daily stock prices
    :type data: pandas.DataFrame

    :param dividend: Paid dividend
    :type dividend: :py:data:`~.finquant.type_definitions.NUMERIC`, default: 0

    :return: A dataframe of cumulative returns of given stock prices.
    :rtype: pandas.DataFrame
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return data.dropna(axis=0, how="any").apply(lambda x: (x - x[0] + dividend) / x[0])


def daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with daily returns (percentage change)

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}`

    :param data: A dataframe of daily stock prices
    :type data: pandas.DataFrame

    :return: A dataframe of daily percentage change of returns of given stock prices.
    :rtype: pandas.DataFrame
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return data.pct_change().dropna(how="all").replace([np.inf, -np.inf], np.nan)


def daily_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with daily log returns

    :math:`R_{\\log} = \\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}
    {\\text{price}_{t_{i-1}}}\\right)`

    :param data: A dataframe of daily stock prices
    :type data: pandas.DataFrame

    :return: A dataframe of daily log returns
    :rtype: pandas.DataFrame
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return np.log(1 + daily_returns(data)).dropna(how="all")


def historical_mean_return(data: SERIES_OR_DATAFRAME, freq: INT = 252) -> pd.Series:
    """Returns the *mean return* based on historical stock price data.

    :param data: A dataframe of daily stock prices
    :type data: :py:data:`~.finquant.type_definitions.SERIES_OR_DATAFRAME`

    :param freq: Number of trading days in a year
    :type freq: :py:data:`~.finquant.type_definitions.INT`, default: 252

    :return: A series of historical mean returns
    :rtype: pandas.Series
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a pandas.DataFrame or pandas.Series")
    return daily_returns(data).mean() * freq
