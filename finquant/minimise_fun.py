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

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param mean_returns: An array of individual expected returns for all stocks
    :type mean_returns: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param cov_matrix: Covariance matrix of returns
    :type cov_matrix: :py:data:`~.finquant.type_definitions.ARRAY_OR_DATAFRAME`

    :return: Annualised volatility
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
    """
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[1]


def negative_sharpe_ratio(
    weights: ARRAY_OR_SERIES,
    mean_returns: ARRAY_OR_SERIES,
    cov_matrix: ARRAY_OR_DATAFRAME,
    risk_free_rate: FLOAT,
) -> FLOAT:
    """Calculates the negative Sharpe ratio of a portfolio

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param mean_returns: An array of individual expected returns for all stocks
    :type mean_returns: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param cov_matrix: Covariance matrix of returns
    :type cov_matrix: :py:data:`~.finquant.type_definitions.ARRAY_OR_DATAFRAME`

    :param risk_free_rate: Risk free rate
    :type risk_free_rate: :py:data:`~.finquant.type_definitions.FLOAT`

    :return: Negative sharpe ratio
    :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
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

    :param weights: An array of weights
    :type weights: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param mean_returns: An array of individual expected returns for all stocks
    :type mean_returns: :py:data:`~.finquant.type_definitions.ARRAY_OR_SERIES`

    :param cov_matrix: Covariance matrix of returns
    :type cov_matrix: :py:data:`~.finquant.type_definitions.ARRAY_OR_DATAFRAME`

    :return: Expected annualised return
    :rtype: :py:data:`~.finquant.type_definitions.NUMERIC`
    """
    return annualised_portfolio_quantities(weights, mean_returns, cov_matrix)[0]
