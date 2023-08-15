"""The module provides functions to compute different kinds of returns of stocks."""


from typing import Any

import numpy as np
import pandas as pd

from finquant.data_types import (
    ARRAY_OR_SERIES,
    FLOAT,
    INT,
    NUMERIC,
    SERIES_OR_DATAFRAME,
)
from finquant.type_utilities import type_validation


def cumulative_returns(data: pd.DataFrame, dividend: NUMERIC = 0) -> pd.DataFrame:
    """Returns DataFrame with cumulative returns

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}
    {\\text{price}_{t_0}}`

    :param data: A dataframe of daily stock prices

    :param dividend: Paid dividend
    :type dividend: :py:data:`~.finquant.data_types.NUMERIC`, default: 0

    :return: A dataframe of cumulative returns of given stock prices.
    """
    # Type validations:
    type_validation(data=data, dividend=dividend)
    data = data.dropna(axis=0, how="any")
    return ((data - data.iloc[0] + dividend) / data.iloc[0]).astype(np.float64)


def daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with daily returns (percentage change)

    :math:`\\displaystyle R = \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}`

    :param data: A dataframe of daily stock prices

    :return: A dataframe of daily percentage change of returns of given stock prices.
    """
    # Type validations:
    type_validation(data=data)
    return (
        data.pct_change()
        .dropna(how="all")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float64)
    )


def weighted_mean_daily_returns(
    data: pd.DataFrame, weights: ARRAY_OR_SERIES[FLOAT]
) -> np.ndarray[FLOAT, Any]:
    """Returns DataFrame with the daily weighted mean returns

    :param data: A dataframe of daily stock prices

    :param weights: An array representing weights
    :type weights: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :return: An array of weighted mean daily percentage change of Returns
    """
    # Type validations:
    type_validation(data=data, weights=weights)
    res: np.ndarray[FLOAT, Any] = np.dot(daily_returns(data), weights)
    return res


def daily_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with daily log returns

    :math:`R_{\\log} = \\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}
    {\\text{price}_{t_{i-1}}}\\right)`

    :param data: A dataframe of daily stock prices

    :return: A dataframe of daily log returns
    """
    # Type validations:
    type_validation(data=data)
    return np.log(1 + daily_returns(data)).dropna(how="all").astype(np.float64)


def historical_mean_return(data: SERIES_OR_DATAFRAME, freq: INT = 252) -> pd.Series:
    """Returns the *mean return* based on historical stock price data.

    :param data: A dataframe of daily stock prices
    :type data: :py:data:`~.finquant.data_types.SERIES_OR_DATAFRAME`

    :param freq: Number of trading days in a year
    :type freq: :py:data:`~.finquant.data_types.INT`, default: 252

    :return: A series of historical mean returns
    """
    # Type validations:
    type_validation(data=data, freq=freq)
    return daily_returns(data).mean() * freq
