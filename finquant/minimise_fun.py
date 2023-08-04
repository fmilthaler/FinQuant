"""This module provides a set of function which can used by
scipy.optimize.minimize in order to find the minimal/optimal value.
"""


from finquant.data_types import ARRAY_OR_DATAFRAME, ARRAY_OR_SERIES, FLOAT, NUMERIC
from finquant.quants import annualised_portfolio_quantities
from finquant.type_utilities import type_validation


def portfolio_volatility(
    weights: ARRAY_OR_SERIES[FLOAT],
    mean_returns: ARRAY_OR_SERIES[FLOAT],
    cov_matrix: ARRAY_OR_DATAFRAME[FLOAT],
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
    # Type validations:
    type_validation(weights=weights, means=mean_returns, cov_matrix=cov_matrix)
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[1]


def negative_sharpe_ratio(
    weights: ARRAY_OR_SERIES[FLOAT],
    mean_returns: ARRAY_OR_SERIES[FLOAT],
    cov_matrix: ARRAY_OR_DATAFRAME[FLOAT],
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
    # Type validations:
    type_validation(
        weights=weights,
        means=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
    )
    sharpe = annualised_portfolio_quantities(
        weights, mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )[2]
    # to find the maximum Sharpe ratio with scipy.optimize.minimize,
    # return the negative of the calculated Sharpe ratio
    return -sharpe


def portfolio_return(
    weights: ARRAY_OR_SERIES[FLOAT],
    mean_returns: ARRAY_OR_SERIES[FLOAT],
    cov_matrix: ARRAY_OR_DATAFRAME[FLOAT],
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
    # Type validations:
    type_validation(weights=weights, means=mean_returns, cov_matrix=cov_matrix)
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[0]
