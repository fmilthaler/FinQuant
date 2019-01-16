'''
This module provides a set of function which can used by scipy.optimize.minimize
in order to find the minimal/optimal value.
'''


import numpy as np
import pandas as pd
from qpy.quants import annualised_portfolio_quantities


def negative_sharpe_ratio(weights, meanReturns, cov_matrix, riskFreeRate):
    '''
    Calculates the negative Sharpe ratio of a portfolio

    Input:
     * weights: numpy.ndarray, weights of the stocks in the portfolio
     * meanReturns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     * cov_matrix: pandas.DataFrame, covariance matrix of returns
     * riskFreeRate: Float (default=0.005), risk free rate
    '''
    sharpe = annualised_portfolio_quantities(weights,
                                             meanReturns,
                                             cov_matrix,
                                             riskFreeRate=riskFreeRate)[2]
    # to find the maximum Sharpe ratio with scipy.optimize.minimize,
    # return the negative of the calculated Sharpe ratio
    return -sharpe
