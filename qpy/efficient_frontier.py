"""
This module facilitates a class EfficientFrontier that can be used to optimise
a portfolio.
"""


import numpy as np
import pandas as pd
import scipy.optimize as sco
import qpy.minimise_fun as min_fun
from qpy.quants import annualised_portfolio_quantities


class EfficientFrontier(object):
    '''
    An object designed to perform optimisations based on the efficient frontier
    of a given portfolio.

    It can find parameters for portfolios with
     - minimum volatility
     - maximum Sharpe ratio
    '''

    def __init__(self, meanReturns, cov_matrix, solver='SLSQP'):
        '''
        Input:
         * meanReturns: pandas.Series, individual expected returns for all
             stocks in the portfolio
         * cov_matrix: pandas.DataFrame, covariance matrix of returns
         * solver: string (default: "SLSQP"), type of solver to use, must be
             one of:
             - 'Nelder-Mead'
             - 'Powell'
             - 'CG'
             - 'BFGS'
             - 'Newton-CG'
             - 'L-BFGS-B'
             - 'TNC'
             - 'COBYLA'
             - 'SLSQP'
             - 'trust-constr'
             - 'dogleg'
             - 'trust-ncg'
             - 'trust-exact'
             - 'trust-krylov'
             all of which are officially supported by scipy.optimize.minimize
        '''
        if (not isinstance(meanReturns, pd.Series)):
            raise ValueError("meanReturns is expected to be a pandas.Series.")
        if (not isinstance(cov_matrix, pd.DataFrame)):
            raise ValueError("cov_matrix is expected to be a pandas.DataFrame")
        supported_solvers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                             'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
                             'trust-constr', 'dogleg', 'trust-ncg',
                             'trust-exact', 'trust-krylov']
        if (not isinstance(solver, str)):
            raise ValueError("solver is expected to be a string.")
        if (solver not in supported_solvers):
            raise ValueError("solver is not supported by scipy.optimize.minimize.")

        # instance variables
        self.meanReturns = meanReturns
        self.cov_matrix = cov_matrix
        self.solver = solver
        self.names = list(meanReturns.index)
        self.num_stocks = len(self.names)

        # set numerical parameters
        bound = (0, 1)
        self.bounds = tuple(bound for stock in range(self.num_stocks))
        self.x0 = np.array(self.num_stocks * [1./self.num_stocks])
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # placeholder for optimised values/weights
        self.weights = None

    def minimum_volatility(self):
        '''
        Finds the portfolio with the minimum volatility.
        '''
        args = (self.meanReturns.values, self.cov_matrix.values)
        result = sco.minimize(min_fun.portfolio_volatility,
                              args=args,
                              x0=self.x0,
                              method=self.solver,
                              bounds=self.bounds,
                              constraints=self.constraints)
        return result

    def maximum_sharpe_ratio(self, riskFreeRate=0.005):
        '''
        Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        Input:
         * riskFreeRate: Float (default: 0.005), the risk free rate as
             required for the Sharpe Ratio
        '''
        if (not isinstance(riskFreeRate, (int, float))):
            raise ValueError("riskFreeRate is required to be an integer or float.")
        args = (self.meanReturns.values, self.cov_matrix.values, riskFreeRate)
        # optimisation
        result = sco.minimize(min_fun.negative_sharpe_ratio,
                              args=args,
                              x0=self.x0,
                              method=self.solver,
                              bounds=self.bounds,
                              constraints=self.constraints)
        # set optimal weights
        self.weights = result['x']
        return pd.DataFrame(self.weights, index=self.names).transpose()


    def properties(self, riskFreeRate=0.005, verbose=True):
        '''
        Calculates and prints out expected annualised return, volatility and
        Sharpe ratio of optimised portfolio.

        Input:
         * riskFreeRate: Float (default=0.005), risk free rate
         * verbose: Boolean (default: True), whether to print out properties
             or not
        '''
        if (self.weights is None):
            raise ValueError("Perform an optimisation first.")

        expectedReturn, volatility, sharpe = annualised_portfolio_quantities(
            self.weights,
            self.meanReturns,
            self.cov_matrix,
            riskFreeRate=riskFreeRate)
        if (verbose):
            print("Expected annual return: {:.3f}".format(expectedReturn))
            print("Annual volatility: {:.3f}".format(volatility))
            print("Sharpe Ratio: {:.3f}".format(sharpe))
        return (expectedReturn, volatility, sharpe)
