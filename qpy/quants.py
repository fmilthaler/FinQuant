'''
Provides functions to compute quantities relevant to financial portfolios,
e.g. a weighted average, which is the expected value/return, a weighted
standard deviation (volatility), and the Sharpe ratio.
'''
import numpy as np


def weightedMean(means, weights):
    '''
    Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the expected return
    of said portfolio.

    Input:
     * means: List/Array of mean/average values
     * weights: List/Array of weights

    Output: Array: (np.sum(means*weights))
    '''
    return np.sum(means * weights)


def weightedStd(cov_matrix, weights):
    '''
    Computes the weighted standard deviation, or volatility of
    a portfolio, which contains several stocks.

    Input:
     * cov_matrix: Array, covariance matrix
     * weights: List/Array of weights

    Output: Array: np.sqrt(np.dot(weights.T,
        np.dot(cov_matrix, weights)))
    '''
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sharpeRatio(expReturn, volatility, riskFreeRate=0.005):
    '''
    Computes the Sharpe Ratio

    Input:
     * expReturn: Float, expected return of a portfolio
     * volatility: Float, volatility of a portfolio
     * riskFreeRate: Float (default=0.005), risk free rate

    Output: Float: (expReturn - riskFreeRate)/float(volatility)
    '''
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
     * weights: List/Array of weights
     * means: List/Array of mean/average values
     * cov_matrix: Array, covariance matrix
     * riskFreeRate: Float (default=0.005), risk free rate
     * freq: Integer (default: 252), number of trading days, default
         value corresponds to trading days in a year
    '''
    expectedReturn = weightedMean(means, weights) * freq
    volatility = weightedStd(cov_matrix, weights) * np.sqrt(freq)
    sharpe = sharpeRatio(expectedReturn, volatility, riskFreeRate)
    return (expectedReturn, volatility, sharpe)
