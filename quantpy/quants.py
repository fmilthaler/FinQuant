'''
Provides functions to compute quantities relevant to financial portfolios,
e.g. a weighted average, which is the expected value/return, a weighted
standard deviation (volatility), and the Sharpe ratio.
'''


import numpy as np
import pandas as pd


def weightedMean(means, weights):
    '''
    Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the expected return
    of said portfolio.

    Input:
     * means: List/Array/pd.Series of mean/average values
     * weights: List/Array/pd.Series of weights

    Output: Array: (np.sum(means*weights))
    '''
    if (not isinstance(weights, (list, pd.Series, np.ndarray))):
            raise ValueError("weights is expected to be a "
                             + "list/pandas.Series, np.array")
    if (not isinstance(means, (list, pd.Series, np.ndarray))):
            raise ValueError("means is expected to be a "
                             + "list/pandas.Series/np.ndarray")
    return np.sum(means * weights)


def weightedStd(cov_matrix, weights):
    '''
    Computes the weighted standard deviation, or volatility of
    a portfolio, which contains several stocks.

    Input:
     * cov_matrix: Array/pandas.DataFrame, covariance matrix
     * weights: List/Array/pd.Series of weights

    Output: Array: np.sqrt(np.dot(weights.T,
        np.dot(cov_matrix, weights)))
    '''
    if (not isinstance(weights, (list, pd.Series, np.ndarray))):
            raise ValueError("weights is expected to be a "
                             + "list/pandas.Series, np.array")
    if (not isinstance(cov_matrix, (np.ndarray, (np.ndarray, pd.DataFrame)))):
            raise ValueError("cov_matrix is expected to be a "
                             + "numpy.array/pandas.DataFrame")
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpeRatio(expReturn, volatility, riskFreeRate=0.005):
    '''
    Computes the Sharpe Ratio

    Input:
     * expReturn: int/float, expected return of a portfolio
     * volatility: int/float, volatility of a portfolio
     * riskFreeRate: int/float (default=0.005), risk free rate

    Output: Float: (expReturn - riskFreeRate)/float(volatility)
    '''
    if (not isinstance(expReturn, (int, float))):
            raise ValueError("expReturn is expected to be an integer "
                             + "or float.")
    if (not isinstance(volatility, (int, float))):
            raise ValueError("volatility is expected to be an integer "
                             + "or float.")
    if (not isinstance(riskFreeRate, (int, float))):
            raise ValueError("riskFreeRate is expected to be an integer "
                             + "or float.")
    return (expReturn - riskFreeRate)/float(volatility)


def annualised_portfolio_quantities(weights,
                                    means,
                                    cov_matrix,
                                    riskFreeRate=0.005,
                                    freq=252):
    '''
    Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    Input:
     * weights: List/Array/pd.Series of weights
     * means: List/Array/pd.Series of mean/average values
     * cov_matrix: Array/pandas.DataFrame, covariance matrix
     * riskFreeRate: Float (default=0.005), risk free rate
     * freq: Integer (default: 252), number of trading days, default
         value corresponds to trading days in a year
    '''
    if (not isinstance(freq, int)):
            raise ValueError("freq is expected to be an integer.")

    expectedReturn = weightedMean(means, weights) * freq
    volatility = weightedStd(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpeRatio(expectedReturn, volatility, riskFreeRate)
    return (expectedReturn, volatility, sharpe)
