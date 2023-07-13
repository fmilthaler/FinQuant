""" This module provides a public class ``Stock`` that holds and calculates quantities of a single stock.
Instances of this class are used in the ``Portfolio`` class (provided in ``finquant.portfolio``).   
Every time a new instance of ``Stock`` is added to ``Portfolio``, the quantities of the portfolio are updated.  
"""

import numpy as np
import pandas as pd
from finquant.returns import historical_mean_return
from finquant.returns import daily_returns


class Stock(object):
    """Object that contains information about a stock/fund.
    To initialise the object, it requires a name, information about
    the stock/fund given as one of the following data structures:

    - ``pandas.Series``
    - ``pandas.DataFrame``

    The investment information can contain as little information as its name,
    and the amount invested in it, the column labels must be ``Name`` and ``Allocation``
    respectively, but it can also contain more information, such as

    - Year
    - Strategy
    - CCY
    - etc.

    It also requires either data, e.g. daily closing prices as a
    ``pandas.DataFrame`` or ``pandas.Series``.
    ``data`` must be given as a ``pandas.DataFrame``, and at least one data column
    is required to containing the closing price, hence it is required to
    contain one column label ``<stock_name> - Adj. Close`` which is used to
    compute the return of investment. However, ``data`` can contain more
    data in additional columns.
    """

    def __init__(self, investmentinfo, data):
        """
        :Input:
         :investmentinfo: ``pandas.DataFrame`` of investment information
         :data: ``pandas.DataFrame`` of stock price
        """
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.data = data
        # compute expected return and volatility of stock
        self.expected_return = self.comp_expected_return()
        self.volatility = self.comp_volatility()
        self.skew = self._comp_skew()
        self.kurtosis = self._comp_kurtosis()
        # beta parameter of stock (CAPM)
        self.beta = None

    # functions to compute quantities
    def comp_daily_returns(self):
        """Computes the daily returns (percentage change).
        See ``finquant.returns.daily_returns``.
        """
        return daily_returns(self.data)

    def comp_expected_return(self, freq=252):
        """Computes the Expected Return of the stock.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :expected_return: Expected Return of stock.
        """
        return historical_mean_return(self.data, freq=freq)

    def comp_volatility(self, freq=252):
        """Computes the Volatility of the stock.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :volatility: Volatility of stock.
        """
        return self.comp_daily_returns().std() * np.sqrt(freq)

    def _comp_skew(self):
        """Computes and returns the skewness of the stock."""
        return self.data.skew().values[0]

    def _comp_kurtosis(self):
        """Computes and returns the Kurtosis of the stock."""
        return self.data.kurt().values[0]

    def comp_beta(self, market_daily_returns: pd.Series) -> float:
        """Compute and return the Beta parameter of the stock.

        :Input:
         :market_daily_returns: ``pd.Series``, daily returns of the market

        :Output:
         :sharpe: ``float``, the Beta parameter of the stock
        """
        cov_mat = np.cov(
            self.comp_daily_returns()[self.name],
            market_daily_returns.to_frame()[market_daily_returns.name],
        )

        beta = cov_mat[0, 1] / cov_mat[1, 1]
        self.beta = beta
        return beta

    def properties(self):
        """Nicely prints out the properties of the stock: Expected Return,
        Volatility, Skewness, Kurtosis as well as the ``Allocation`` (and other
        information provided in investmentinfo.)
        """
        # nicely printing out information and quantities of the stock
        string = "-" * 50
        string += "\nStock: {}".format(self.name)
        string += "\nExpected Return:{:0.3f}".format(self.expected_return.values[0])
        string += "\nVolatility: {:0.3f}".format(self.volatility.values[0])
        string += "\nSkewness: {:0.5f}".format(self.skew)
        string += "\nKurtosis: {:0.5f}".format(self.kurtosis)
        string += "\nInformation:"
        string += "\n" + str(self.investmentinfo.to_frame().transpose())
        string += "\n"
        string += "-" * 50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about " + str(self.name) + "."
        return string
