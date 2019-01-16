'''
This module provides a set of function which can used by
scipy.optimize.minimize in order to find the minimal/optimal value.
'''


from quantpy.quants import annualised_portfolio_quantities


def portfolio_volatility(weights, mean_returns, cov_matrix):
    '''
    Calculates the negative Sharpe ratio of a portfolio

    Input:
     * weights: numpy.ndarray, weights of the stocks in the portfolio
     * meanReturns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     * cov_matrix: pandas.DataFrame, covariance matrix of returns
     * riskFreeRate: Float (default=0.005), risk free rate
    '''
    return annualised_portfolio_quantities(weights,
                                           mean_returns,
                                           cov_matrix)[1]


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


def portfolio_return(weights, mean_returns, cov_matrix):
    '''
    Calculates the expected annualised return of a portfolio

    Input:
     * weights: numpy.ndarray, weights of the stocks in the portfolio
     * meanReturns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     * cov_matrix: pandas.DataFrame, covariance matrix of returns
    '''
    return annualised_portfolio_quantities(weights,
                                           mean_returns,
                                           cov_matrix)[0]
