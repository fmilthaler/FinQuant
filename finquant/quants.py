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

    :Input:
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted mu: ``numpy.float64``: ``np.sum(means*weights)``
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

    :Input:
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted sigma: ``numpy.float64``:
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

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :volatility: ``int``/``float``, Volatility of a portfolio
     :risk_free_rate: ``float`` (default= ``0.005``), risk free rate

    :Output:
     :sharpe ratio: ``float`` ``(exp_return - risk_free_rate)/float(volatility)``
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
    """Computes the Sortino Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :downs_risk: ``int``/``float``, Downside Risk of a portfolio
     :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate

    :Output:
     :sortino ratio: ``float``/``NaN`` ``(exp_return - risk_free_rate)/float(downside_risk)``.
     Can be ``NaN`` if ``downside_risk`` is zero
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


def downside_risk(
    data: pd.DataFrame, weights: ARRAY_OR_SERIES[FLOAT], risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    """Computes the downside risk (target downside deviation of returns).

    :Input:
      :data: ``pandas.DataFrame`` with daily stock prices
      :weights: ``numpy.ndarray``/``pd.Series`` of weights
      :risk_free_rate: ``int``/``float`` (default=``0.005``), risk free rate

    :Output:
      :downside_risk: ``float``, target downside deviation
    """
    # Type validations:
    type_validation(data=data, weights=weights, risk_free_rate=risk_free_rate)
    wtd_daily_mean = weighted_mean_daily_returns(data, weights)
    return float(np.sqrt(np.mean(np.minimum(0, wtd_daily_mean - risk_free_rate) ** 2)))


def value_at_risk(
    investment: NUMERIC, mu: FLOAT, sigma: FLOAT, conf_level: FLOAT = 0.95
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
