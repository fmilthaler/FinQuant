"""The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""


from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from finquant.data_types import ARRAY_OR_DATAFRAME, ARRAY_OR_SERIES, FLOAT, INT, NUMERIC
from finquant.returns import weighted_mean_daily_returns
from finquant.type_utilities import type_validation


def weighted_mean(
    means: ARRAY_OR_SERIES[FLOAT], weights: ARRAY_OR_SERIES[FLOAT]
) -> FLOAT:
    """Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the Expected Return
    of said portfolio.

    :param means: An array representing mean/average values
    :type means: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :param weights: An array representing weights
    :type weights: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: The weighted mean as a floating point number: ``np.sum(means*weights)``
    """
    # Type validations:
    type_validation(means=means, weights=weights)
    weighted_mu: FLOAT = float(np.sum(means * weights))
    return weighted_mu


def weighted_std(
    cov_matrix: ARRAY_OR_DATAFRAME[FLOAT], weights: ARRAY_OR_SERIES[FLOAT]
) -> FLOAT:
    """Computes the weighted standard deviation, or Volatility of
    a portfolio, which contains several stocks.

    :param cov_matrix: Covariance matrix
    :type cov_matrix: :py:data:`~.finquant.data_types.ARRAY_OR_DATAFRAME`

    :param weights: An array representing weights
    :type weights: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Weighted sigma (standard deviation) as a floating point number:
        ``np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))``
    """
    # Type validations:
    type_validation(cov_matrix=cov_matrix, weights=weights)
    weighted_sigma: FLOAT = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_sigma


def sharpe_ratio(
    exp_return: FLOAT, volatility: FLOAT, risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the Sharpe Ratio

    :param exp_return: Expected Return of a portfolio
    :type exp_return: :py:data:`~.finquant.data_types.FLOAT`

    :param volatility: Volatility of a portfolio
    :type volatility: :py:data:`~.finquant.data_types.FLOAT`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.data_types.FLOAT`, default: 0.005

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Sharpe Ratio as a floating point number:
        ``(exp_return-risk_free_rate)/float(volatility)``
    """
    # Type validations:
    type_validation(
        expected_return=exp_return, volatility=volatility, risk_free_rate=risk_free_rate
    )
    res_sharpe_ratio: FLOAT = (exp_return - risk_free_rate) / float(volatility)
    return res_sharpe_ratio


def sortino_ratio(
    exp_return: FLOAT, downs_risk: FLOAT, risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the Sortino Ratio.

    :param exp_return: Expected Return of a portfolio
    :type exp_return: :py:data:`~.finquant.data_types.FLOAT`

    :param downs_risk: Downside Risk of a portfolio
    :type exp_return: :py:data:`~.finquant.data_types.FLOAT`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.data_types.FLOAT`, default: 0.005

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Sortino Ratio as a floating point number:
        ``(exp_return - risk_free_rate) / float(downs_risk)``
    """
    # Type validations:
    type_validation(
        expected_return=exp_return,
        downside_risk=downs_risk,
        risk_free_rate=risk_free_rate,
    )
    if float(downs_risk) == 0:
        return np.nan
    else:
        return (exp_return - risk_free_rate) / float(downs_risk)


def treynor_ratio(
    exp_return: FLOAT, beta: FLOAT, risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the Treynor Ratio.

    :param exp_return: Expected Return of a portfolio
    :type exp_return: :py:data:`~.finquant.data_types.FLOAT`

    :param beta: Beta parameter of a portfolio
    :type beta: :py:data:`~.finquant.data_types.FLOAT`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.data_types.FLOAT`, default: 0.005

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Treynor Ratio as a floating point number:
        ``(exp_return - risk_free_rate) / float(beta)``
    """
    # Type validations:
    type_validation(
        expected_return=exp_return,
        beta_parameter=beta,
        risk_free_rate=risk_free_rate,
    )
    res_treynor_ratio: FLOAT = (exp_return - risk_free_rate) / float(beta)
    return res_treynor_ratio


def downside_risk(
    data: pd.DataFrame, weights: ARRAY_OR_SERIES[FLOAT], risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the downside risk (target downside deviation of returns).

    :param data: A dataframe of daily stock prices

    :param weights: Downside Risk of a portfolio
    :type weights: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.data_types.FLOAT`, default: 0.005

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Target downside deviation
        ``np.sqrt(np.mean(np.minimum(0, wtd_daily_mean - risk_free_rate) ** 2))``
    """
    # Type validations:
    type_validation(data=data, weights=weights, risk_free_rate=risk_free_rate)
    wtd_daily_mean = weighted_mean_daily_returns(data, weights)
    return float(np.sqrt(np.mean(np.minimum(0, wtd_daily_mean - risk_free_rate) ** 2)))


def value_at_risk(
    investment: NUMERIC, mu: FLOAT, sigma: FLOAT, conf_level: FLOAT = 0.95
) -> FLOAT:
    """Computes and returns the expected value at risk of an investment/assets.

    :param investment: Total value of the investment
    :type investment: :py:data:`~.finquant.data_types.NUMERIC`

    :param mu: Average/mean return of the investment
    :type mu: :py:data:`~.finquant.data_types.FLOAT`

    :param sigma: Standard deviation of the investment
    :type sigma: :py:data:`~.finquant.data_types.FLOAT`

    :param conf_level: Confidence level of the VaR
    :type conf_level: :py:data:`~.finquant.data_types.FLOAT`, default: 0.95

    :rtype: :py:data:`~.finquant.data_types.FLOAT`
    :return: Value at Risk (VaR) of the investment: ``investment*(mu-sigma*norm.ppf(1-conf_level))``
    """
    # Type validations:
    type_validation(investment=investment, mu=mu, sigma=sigma, conf_level=conf_level)
    if conf_level >= 1 or conf_level <= 0:
        raise ValueError("confidence level is expected to be between 0 and 1.")
    res_value_at_risk: FLOAT = investment * (mu - sigma * norm.ppf(1 - conf_level))
    return res_value_at_risk


def annualised_portfolio_quantities(
    weights: ARRAY_OR_SERIES[FLOAT],
    means: ARRAY_OR_SERIES[FLOAT],
    cov_matrix: ARRAY_OR_DATAFRAME[FLOAT],
    risk_free_rate: FLOAT = 0.005,
    freq: INT = 252,
) -> Tuple[FLOAT, FLOAT, FLOAT]:
    """Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :param means: An array of mean/average values
    :type means: :py:data:`~.finquant.data_types.ARRAY_OR_SERIES`

    :param cov_matrix: Covariance matrix
    :type cov_matrix: :py:data:`~.finquant.data_types.ARRAY_OR_DATAFRAME`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.data_types.FLOAT`, default: 0.005

    :param freq: Number of trading days in a year
    :type freq: :py:data:`~.finquant.data_types.INT`, default: 252

    :rtype: Tuple[:py:data:`~.finquant.data_types.FLOAT`,
        :py:data:`~.finquant.data_types.FLOAT`,
        :py:data:`~.finquant.data_types.FLOAT`]
    :return: Tuple of Expected Return, Volatility, Sharpe Ratio
    """
    # Type validations:
    type_validation(
        weights=weights,
        means=means,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        freq=freq,
    )
    expected_return = weighted_mean(means, weights) * freq
    volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)
    return (expected_return, volatility, sharpe)
