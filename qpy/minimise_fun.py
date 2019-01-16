import numpy as np
import pandas as pd
from qpy.quants import annualised_portfolio_quantities


def negative_sharpe_ratio(weights, meanReturns, cov_matrix, riskFreeRate):
    sharpe = annualised_portfolio_quantities(weights,
                                             meanReturns,
                                             cov_matrix,
                                             riskFreeRate=riskFreeRate)[2]
    # to find the maximum Sharpe ratio with scipy.optimize.minimize,
    # return the negative of the calculated Sharpe ratio
    return -sharpe
