"""Provides functions to compute quantities relevant to financial portfolios,
e.g. a weighted average, which is the expected value/return, a weighted
standard deviation (volatility), and the Sharpe ratio.
"""


import numpy as np
import pandas as pd


def weighted_mean(means, weights):
    """Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the expected return
    of said portfolio.

    :Input:
     :means: Array/pd.Series of mean/average values
     :weights: Array/pd.Series of weights

    :Output:
     :weighted mu: numpy.ndarray: (np.sum(means*weights))
    """
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a list/pandas.Series, np.array")
    if not isinstance(means, (pd.Series, np.ndarray)):
        raise ValueError("means is expected to be a list/pandas.Series/np.ndarray")
    return np.sum(means * weights)


def weighted_std(cov_matrix, weights):
    """Computes the weighted standard deviation, or volatility of
    a portfolio, which contains several stocks.

    :Input:
     :cov_matrix: Array/pandas.DataFrame, covariance matrix
     :weights: List/Array/pd.Series of weights

    :Output:
     :weighted sigma: numpy.ndarray:
         np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    """
    if not isinstance(weights, (list, pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a list/pandas.Series, np.array")
    if not isinstance(cov_matrix, (np.ndarray, (np.ndarray, pd.DataFrame))):
        raise ValueError(
            "cov_matrix is expected to be a numpy.ndarray/pandas.DataFrame"
        )
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpe_ratio(exp_return, volatility, risk_free_rate=0.005):
    """Computes the Sharpe Ratio

    :Input:
     :exp_return: int/float, expected return of a portfolio
     :volatility: int/float, volatility of a portfolio
     :risk_free_rate: int/float (default=0.005), risk free rate

    :Output:
     :sharpe ratio: float (exp_return - risk_free_rate)/float(volatility)
    """
    if not isinstance(exp_return, (int, float, np.int64, np.float64)):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(volatility, (int, float, np.int64, np.float64)):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(risk_free_rate, (int, float, np.int64, np.float64)):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    return (exp_return - risk_free_rate) / float(volatility)


def annualised_portfolio_quantities(
    weights, means, cov_matrix, risk_free_rate=0.005, freq=252
):
    """Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    :Input:
     :weights: List/Array/pd.Series of weights
     :means: List/Array/pd.Series of mean/average values
     :cov_matrix: Array/pandas.DataFrame, covariance matrix
     :risk_free_rate: Float (default=0.005), risk free rate
     :freq: Integer (default: 252), number of trading days, default
         value corresponds to trading days in a year

    :Output:
     :(expected return, volatility, sharpe ratio): tuple of those
         three quantities
    """
    if not isinstance(freq, int):
        raise ValueError("freq is expected to be an integer.")
    expected_return = weighted_mean(means, weights) * freq
    volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)
    return (expected_return, volatility, sharpe)
