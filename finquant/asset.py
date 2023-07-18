"""This module provides a public class ``Market`` that holds and calculates quantities of a market index"""

import numpy as np
import pandas as pd

from finquant.returns import daily_returns, historical_mean_return


class Market:
    """Object that contains information about a market index.
    To initialise the object, it requires a name and information about
    the index given as ``pandas.Series`` data structure.
    """

    def __init__(self, data: pd.Series) -> None:
        """
        :Input:
         :data: ``pandas.Series`` of market index prices
        """
        self.name = data.name
        self.data = data
        # compute expected return and volatility of market index
        self.expected_return = self.comp_expected_return()
        self.volatility = self.comp_volatility()
        self.skew = self._comp_skew()
        self.kurtosis = self._comp_kurtosis()
        self.daily_returns = self.comp_daily_returns()

    # functions to compute quantities
    def comp_daily_returns(self) -> pd.Series:
        """Computes the daily returns (percentage change) of the market index.
        See ``finance_portfolio.returns.daily_returns``.
        """
        return daily_returns(self.data)

    def comp_expected_return(self, freq=252) -> float:
        """Computes the Expected Return of the market index.
        See ``finance_portfolio.returns.historical_mean_return``.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :expected_return: Expected Return of market index.
        """
        return historical_mean_return(self.data, freq=freq)

    def comp_volatility(self, freq=252) -> float:
        """Computes the Volatility of the market index.

        :Input:
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year

        :Output:
         :volatility: volatility of market index.
        """
        return self.comp_daily_returns().std() * np.sqrt(freq)

    def _comp_skew(self) -> float:
        """Computes and returns the skewness of the market index."""
        return self.data.skew()

    def _comp_kurtosis(self) -> float:
        """Computes and returns the Kurtosis of the market index."""
        return self.data.kurt()

    def properties(self):
        """Nicely prints out the properties of the market index:
        Expected Return, Volatility, Skewness, and Kurtosis.
        """
        # nicely printing out information and quantities of market index
        string = "-" * 50
        string += f"\nMarket index: {self.name}"
        string += f"\nExpected Return:{self.expected_return:0.3f}"
        string += f"\nVolatility: {self.volatility:0.3f}"
        string += f"\nSkewness: {self.skew:0.5f}"
        string += f"\nKurtosis: {self.kurtosis:0.5f}"
        string += "-" * 50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about market index " + str(self.name) + "."
        return string
