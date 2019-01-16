"""
This module facilitates a class EfficientFrontier that can be used to optimise
a portfolio.
"""


import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pylab as plt
import qpy.minimise_fun as min_fun
from qpy.quants import annualised_portfolio_quantities


class EfficientFrontier(object):
    '''
    An object designed to perform optimisations based on the efficient frontier
    of a given portfolio.

    It can find parameters for portfolios with
     - minimum volatility
     - maximum Sharpe ratio
     - minimum volatility for a given target return
     - maximum Sharpe ratio for a given target volatility
    '''

    def __init__(self, meanReturns, cov_matrix, method='SLSQP'):
        '''
        Input:
         * meanReturns: pandas.Series, individual expected returns for all
             stocks in the portfolio
         * cov_matrix: pandas.DataFrame, covariance matrix of returns
         * method: string (default: "SLSQP"), type of solver method to use,
             must be one of:
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
        supported_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
                             'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
                             'trust-constr', 'dogleg', 'trust-ncg',
                             'trust-exact', 'trust-krylov']
        if (not isinstance(method, str)):
            raise ValueError("method is expected to be a string.")
        if (method not in supported_methods):
            raise ValueError("method is not supported by "
                             + "scipy.optimize.minimize.")

        # instance variables
        self.meanReturns = meanReturns
        self.cov_matrix = cov_matrix
        self.method = method
        self.names = list(meanReturns.index)
        self.num_stocks = len(self.names)

        # set numerical parameters
        bound = (0, 1)
        self.bounds = tuple(bound for stock in range(self.num_stocks))
        self.x0 = np.array(self.num_stocks * [1./self.num_stocks])
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # placeholder for optimised values/weights
        self.weights = None
        self.df_weights = None
        self.efrontier = None

    def minimum_volatility(self):
        '''
        Finds the portfolio with the minimum volatility.
        '''
        args = (self.meanReturns.values, self.cov_matrix.values)
        # optimisation
        result = sco.minimize(min_fun.portfolio_volatility,
                              args=args,
                              x0=self.x0,
                              method=self.method,
                              bounds=self.bounds,
                              constraints=self.constraints)
        # set optimal weights
        self.weights = result['x']
        self.df_weights = self.dataframe_weights(self.weights)
        return self.df_weights

    def maximum_sharpe_ratio(self, riskFreeRate=0.005):
        '''
        Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        Input:
         * riskFreeRate: Float (default: 0.005), the risk free rate as
             required for the Sharpe Ratio
        '''
        if (not isinstance(riskFreeRate, (int, float))):
            raise ValueError("riskFreeRate is required to be an integer "
                             + "or float.")
        args = (self.meanReturns.values, self.cov_matrix.values, riskFreeRate)
        # optimisation
        result = sco.minimize(min_fun.negative_sharpe_ratio,
                              args=args,
                              x0=self.x0,
                              method=self.method,
                              bounds=self.bounds,
                              constraints=self.constraints)
        # set optimal weights
        self.weights = result['x']
        self.df_weights = self.dataframe_weights(self.weights)
        return self.df_weights

    def efficient_return(self, target, save_weights=True):
        '''
        Finds the portfolio with the minimum volatility for a given target
        return.

        Input:
         * target: Float, the target return of the optimised portfolio.
         * save_weights: Boolean (default: True), whether to save the optimised
             weights in the instance variable weights (and df_weights). Useful
             for the case of computing the efficient frontier after doing a
             optimisation, else the optimal weights would be overwritten by the
             efficient frontier computations.

        Output:
         * df_weights/weights:
          - if "save_weights" is True: returning pandas.DataFrame of weights
          - if "save_weights" is False: returning numpy.ndarray of weights
        '''
        if (not isinstance(target, (int, float))):
            raise ValueError("target is required to be an integer or float.")
        args = (self.meanReturns.values, self.cov_matrix.values)
        # here we have an additional constraint:
        constraints = (self.constraints,
                       {'type': 'eq',
                        'fun': lambda x: min_fun.portfolio_return(
                            x, self.meanReturns, self.cov_matrix) - target})
        # optimisation
        result = sco.minimize(min_fun.portfolio_volatility,
                              args=args,
                              x0=self.x0,
                              method=self.method,
                              bounds=self.bounds,
                              constraints=constraints)
        # set optimal weights
        if (save_weights):
            self.weights = result['x']
            self.df_weights = self.dataframe_weights(self.weights)
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result['x']

    def efficient_volatility(self, target, riskFreeRate=0.005):
        '''
        Finds the portfolio with the maximum Sharpe ratio for a given
        target volatility.

        Input:
         * target: Float, the target volatility of the optimised portfolio.
         * riskFreeRate: Float (default: 0.005), the risk free rate as
             required for the Sharpe Ratio
        '''
        if (not isinstance(target, (int, float))):
            raise ValueError("target is required to be an integer or float.")
        args = (self.meanReturns.values, self.cov_matrix.values, riskFreeRate)
        # here we have an additional constraint:
        constraints = (self.constraints,
                       {'type': 'eq',
                        'fun': lambda x: min_fun.portfolio_volatility(
                            x, self.meanReturns, self.cov_matrix) - target})
        # optimisation
        result = sco.minimize(min_fun.negative_sharpe_ratio,
                              args=args,
                              x0=self.x0,
                              method=self.method,
                              bounds=self.bounds,
                              constraints=constraints)
        # set optimal weights
        self.weights = result['x']
        self.df_weights = self.dataframe_weights(self.weights)
        return self.df_weights

    def efficient_frontier(self, targets):
        '''
        Gets portfolios for a range of given target returns.
        Results in the Efficient Frontier.

        Input:
         * targets: list of floats, range of target returns

        Output:
         * array of (volatility, return) values
        '''
        efrontier = []
        for target in targets:
            weights = self.efficient_return(target, save_weights=False)
            efrontier.append(
                [annualised_portfolio_quantities(weights,
                                                 self.meanReturns,
                                                 self.cov_matrix)[1],
                 target])
        self.efrontier = np.array(efrontier)
        return efrontier

    def plot_efrontier(self, show=True):
        '''
        Plots the Efficient Frontier.

        Input:
         * show: Boolean (default: True) whether to do plt.show()
             or not. Useful if more data should be plotted in the same
             figure.
        '''
        plt.plot(self.efrontier[:, 0],
                 self.efrontier[:, 1],
                 linestyle='-.',
                 color='black',
                 lw=2,
                 label='Efficient Frontier')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        if (show):
            plt.show()

    def dataframe_weights(self, weights):
        '''
        Generates and returns a pandas.DataFrame from given array weights.

        Input:
         * weights: numpy.ndarray, weights of the stock of the portfolio

        Output:
         * pandas.DataFrame(self.weights,
             index=self.names,
             columns=['Allocation'])
        '''
        return pd.DataFrame(weights, index=self.names, columns=['Allocation'])

    def properties(self, riskFreeRate=0.005, verbose=False):
        '''
        Calculates and prints out expected annualised return, volatility and
        Sharpe ratio of optimised portfolio.

        Input:
         * riskFreeRate: Float (default=0.005), risk free rate
         * verbose: Boolean (default: False), whether to print out properties
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
            string = "Expected annual return: {:.3f}".format(expectedReturn)
            string += "\nAnnual volatility: {:.3f}".format(volatility)
            string += "\nSharpe Ratio: {:.3f}".format(sharpe)
            string += "\nOptimal weights:"
            string += "\n"+str(self.df_weights.transpose())
            print(string)
        return (expectedReturn, volatility, sharpe)
