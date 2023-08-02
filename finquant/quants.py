"""The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""


from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from finquant.returns import weighted_mean_daily_returns
from finquant.type_definitions import (
    ARRAY_OR_DATAFRAME,
    ARRAY_OR_SERIES,
    FLOAT,
    INT,
    NUMERIC,
)


def weighted_mean(means: ARRAY_OR_SERIES, weights: ARRAY_OR_SERIES) -> FLOAT:
    """Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the Expected Return
    of said portfolio.

    :Input:
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted mu: ``numpy.float64``: ``np.sum(means*weights)``
    """
    if not isinstance(weights, (np.ndarray, pd.Series)):
        raise ValueError("weights is expected to be a numpy.ndarray/pandas.Series")
    if not isinstance(means, (np.ndarray, pd.Series)):
        raise ValueError("means is expected to be a numpy.ndarray/pandas.Series")
    weighted_mu: FLOAT = np.sum(means * weights)
    return weighted_mu


def weighted_std(cov_matrix: ARRAY_OR_DATAFRAME, weights: ARRAY_OR_SERIES) -> FLOAT:
    """Computes the weighted standard deviation, or Volatility of
    a portfolio, which contains several stocks.

    :Input:
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted sigma: ``numpy.float64``:
         ``np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))``
    """
    if not isinstance(weights, (np.ndarray, pd.Series)):
        raise ValueError("weights is expected to be a numpy.ndarray/pandas.Series")
    if not isinstance(cov_matrix, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            "cov_matrix is expected to be a numpy.ndarray/pandas.DataFrame"
        )
    weighted_sigma: FLOAT = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_sigma


def sharpe_ratio(
    exp_return: NUMERIC, volatility: NUMERIC, risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the Sharpe Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :volatility: ``int``/``float``, Volatility of a portfolio
     :risk_free_rate: ``float`` (default= ``0.005``), risk free rate

    :Output:
     :sharpe ratio: ``float`` ``(exp_return - risk_free_rate)/float(volatility)``
    """
    if not isinstance(exp_return, (np.number, int, float)):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(volatility, (np.number, int, float)):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(risk_free_rate, (np.number, int, float)):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    res_sharpe_ratio: FLOAT = (exp_return - risk_free_rate) / float(volatility)
    return res_sharpe_ratio


def sortino_ratio(exp_return, downside_risk, risk_free_rate=0.005):
    """Computes the Sortino Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :downside_risk: ``int``/``float``, Downside Risk of a portfolio
     :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate

    :Output:
     :sortino ratio: ``float``/``NaN`` ``(exp_return - risk_free_rate)/float(downside_risk)``.
     Can be ``NaN`` if ``downside_risk`` is zero
    """
    if not isinstance(
        exp_return, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(
        downside_risk, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(
        risk_free_rate, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    if float(downside_risk) == 0:
        return np.nan
    else:
        return (exp_return - risk_free_rate) / float(downside_risk)


def downside_risk(data: pd.DataFrame, weights, risk_free_rate=0.005) -> float:
    """Computes the downside risk (target downside deviation of returns).

    :Input:
      :data: ``pandas.DataFrame`` with daily stock prices
      :weights: ``numpy.ndarray``/``pd.Series`` of weights
      :risk_free_rate: ``int``/``float`` (default=``0.005``), risk free rate

    :Output:
      :downside_risk: ``float``, target downside deviation
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data is expected to be a Pandas.DataFrame.")
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a pandas.Series/np.ndarray.")
    if not isinstance(
        risk_free_rate, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("risk_free_rate is expected to be an integer or float.")

    wtd_daily_mean = weighted_mean_daily_returns(data, weights)
    return np.sqrt(np.mean(np.minimum(0, wtd_daily_mean - risk_free_rate) ** 2))


def value_at_risk(
    investment: NUMERIC, mu: NUMERIC, sigma: NUMERIC, conf_level: FLOAT = 0.95
) -> FLOAT:
    """Computes and returns the expected value at risk of an investment/assets.

    :Input:
     :investment: ``float``/``int``, total value of the investment
     :mu: ``float``/``int`` average/mean return of the investment
     :sigma: ``float``/``int`` standard deviation of the investment
     :conf_level: ``float`` (default= ``0.95``), confidence level of the VaR

    :Output:
     :Value at Risk: ``float``, VaR of the investment
    """
    if not isinstance(investment, (np.number, int, float)):
        raise ValueError("investment is expected to be an integer or float.")
    if not isinstance(mu, (np.number, int, float)):
        raise ValueError("mu is expected to be an integer or float")
    if not isinstance(sigma, (np.number, int, float)):
        raise ValueError("sigma is expected to be an integer or float")
    if not isinstance(conf_level, (float, np.floating)):
        raise ValueError("confidence level is expected to be a float.")
    if conf_level >= 1 or conf_level <= 0:
        raise ValueError("confidence level is expected to be between 0 and 1.")
    res_value_at_risk: FLOAT = investment * (mu - sigma * norm.ppf(1 - conf_level))
    return res_value_at_risk


def annualised_portfolio_quantities(
    weights: ARRAY_OR_SERIES,
    means: ARRAY_OR_SERIES,
    cov_matrix: ARRAY_OR_DATAFRAME,
    risk_free_rate: FLOAT = 0.005,
    freq: INT = 252,
) -> Tuple[NUMERIC, FLOAT, FLOAT]:
    """Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    :Input:
     :weights: ``numpy.ndarray``/``pd.Series`` of weights
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :risk_free_rate: ``float`` (default= ``0.005``), risk free rate
     :freq: ``int`` (default= ``252``), number of trading days, default
         value corresponds to trading days in a year

    :Output:
     :(Expected Return, Volatility, Sharpe Ratio): tuple of those
         three quantities
    """
    if not isinstance(freq, int):
        raise ValueError("freq is expected to be an integer.")
    expected_return = weighted_mean(means, weights) * freq
    volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)
    return (expected_return, volatility, sharpe)
