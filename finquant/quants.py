"""The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""


from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

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

    :param means: An array of mean/average values
    :type means: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :return: The weighted mean as a floating point number: ``np.sum(means*weights)``
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
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

    :param cov_matrix: An array representing the covariance matrix
    :type cov_matrix: :py:data:`~.finquant.type_definitions.ARRAY_OR_DATAFRAME`

    :param weights: An array representing weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :return: Weighted sigma (standard deviation) as a floating point number:
        ``np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))``
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
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

    :param exp_return: Expected Return of a portfolio
    :type exp_return: :py:data:`~.finquant.type_definitions.NUMERIC`

    :param volatility: Volatility of a portfolio
    :type volatility: :py:data:`~.finquant.type_definitions.NUMERIC`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.type_definitions.FLOAT`, default: 0.005

    :return: Sharpe Ratio as a floating point number:
        ``(exp_return-risk_free_rate)/float(volatility)``
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
    """
    if not isinstance(exp_return, (np.number, int, float)):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(volatility, (np.number, int, float)):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(risk_free_rate, (np.number, int, float)):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    res_sharpe_ratio: FLOAT = (exp_return - risk_free_rate) / float(volatility)
    return res_sharpe_ratio


def value_at_risk(
    investment: NUMERIC, mu: NUMERIC, sigma: NUMERIC, conf_level: FLOAT = 0.95
) -> FLOAT:
    """Computes and returns the expected value at risk of an investment/assets.

    :param investment: Total value of the investment
    :type investment: :py:data:`~.finquant.type_definitions.NUMERIC`

    :param mu: Average/mean return of the investment
    :type mu: :py:data:`~.finquant.type_definitions.NUMERIC`

    :param sigma: Standard deviation of the investment
    :type sigma: :py:data:`~.finquant.type_definitions.NUMERIC`

    :param conf_level: Confidence level of the VaR
    :type conf_level: :py:data:`~.finquant.type_definitions.FLOAT`, default: 0.95

    :return: Value at Risk (VaR) of the investment: ``investment*(mu-sigma*norm.ppf(1-conf_level))``
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
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

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param means: An array of mean/average values
    :type means: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param cov_matrix: Covariance matrix
    :type cov_matrix: :py:data:`~.finquant.type_definitions.ARRAY_OR_DATAFRAME`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.type_definitions.FLOAT`, default: 0.005

    :param freq: Number of trading days in a year
    :type freq: :py:data:`~.finquant.type_definitions.INT`, default: 252

    :return: Tuple of Expected Return, Volatility, Sharpe Ratio
    :rtype: Tuple[:py:data:`~.finquant.type_definitions.NUMERIC`,
        :py:data:`~.finquant.type_definitions.FLOAT`,
        :py:data:`~.finquant.type_definitions.FLOAT`]
    """
    if not isinstance(freq, int):
        raise ValueError("freq is expected to be an integer.")
    expected_return = weighted_mean(means, weights) * freq
    volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)
    return (expected_return, volatility, sharpe)
