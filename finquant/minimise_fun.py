"""This module provides a set of function which can used by
scipy.optimize.minimize in order to find the minimal/optimal value.
"""


from finquant.quants import annualised_portfolio_quantities
from finquant.type_definitions import (
    ARRAY_OR_DATAFRAME,
    ARRAY_OR_SERIES,
    FLOAT,
    NUMERIC,
)


def portfolio_volatility(
    weights: ARRAY_OR_SERIES,
    mean_returns: ARRAY_OR_DATAFRAME,
    cov_matrix: ARRAY_OR_DATAFRAME,
) -> FLOAT:
    """Calculates the volatility of a portfolio

    :Input:
     :weights: numpy.ndarray, weights of the stocks in the portfolio
     :mean_returns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     :cov_matrix: pandas.DataFrame, covariance matrix of returns

    Output:
     :volatility: annualised volatility
    """
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[1]


def negative_sharpe_ratio(
    weights: ARRAY_OR_SERIES,
    mean_returns: ARRAY_OR_SERIES,
    cov_matrix: ARRAY_OR_DATAFRAME,
    risk_free_rate: FLOAT,
) -> FLOAT:
    """Calculates the negative Sharpe ratio of a portfolio

    :Input:
     :weights: numpy.ndarray, weights of the stocks in the portfolio
     :mean_returns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     :cov_matrix: pandas.DataFrame, covariance matrix of returns
     :risk_free_rate: Float (default=0.005), risk free rate

    Output:
     :sharpe: sharpe ratio * (-1)
    """
    sharpe = annualised_portfolio_quantities(
        weights, mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )[2]
    # to find the maximum Sharpe ratio with scipy.optimize.minimize,
    # return the negative of the calculated Sharpe ratio
    return -sharpe


def portfolio_return(
    weights: ARRAY_OR_SERIES,
    mean_returns: ARRAY_OR_SERIES,
    cov_matrix: ARRAY_OR_DATAFRAME,
) -> NUMERIC:
    """Calculates the expected annualised return of a portfolio

    :Input:
     :weights: numpy.ndarray, weights of the stocks in the portfolio
     :mean_returns: pandas.Series, individual expected returns for all stocks
         in the portfolio
     :cov_matrix: pandas.DataFrame, covariance matrix of returns

    Output:
     :return: expected annualised return
    """
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[0]
