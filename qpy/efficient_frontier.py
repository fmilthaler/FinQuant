import numpy as np
import pandas as pd


class EfficientFrontier(object):
    def __init__(self, meanReturns, cov_matrix, solver='SLSQP'):
        if (not isinstance(meanReturns, pd.Series)):
            raise ValueError("meanReturns is expected to be a pandas.Series.")
        if (not isinstance(cov_matrix, pd.DataFrame)):
            raise ValueError("cov_matrix is expected to be a pandas.DataFrame")
        if (not isinstance(solver, str)):
            raise ValueError("solver is expected to be a string.")
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
