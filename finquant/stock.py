"""
This module provides a public class ``Stock`` that represents a single stock or fund.
Instances of this class are used within the ``Portfolio`` class (provided in ``finquant.portfolio``).

The ``Stock`` class is designed to hold and calculate quantities related to a single stock or fund.
To initialize an instance of ``Stock``, it requires the following information:

    - ``investmentinfo``: Information about the stock or fund provided as a ``pandas.DataFrame``.
        The required column labels are ``Name`` and ``Allocation`` for the stock/fund name and allocation,
        respectively. However, the DataFrame can contain more information beyond these columns,
        such as year, strategy, currency (CCY), etc.

    - ``data``: Historical price data for the stock or fund provided as a ``pandas.DataFrame``.
        The data must contain `<stock_name> - Adj. Close`, which represents the closing price used to
        compute the return on investment. The DataFrame can contain additional columns as well.

The ``Stock`` class computes various quantities related to the stock or fund, such as expected return,
volatility, skewness, and kurtosis. It also provides functionality to calculate the beta parameter
of the stock using the CAPM (Capital Asset Pricing Model).

The ``Stock`` class inherits from the ``Asset`` class in ``finquant.asset``, which provides common
functionality and attributes for financial assets.

"""

import numpy as np
import pandas as pd
from finquant.asset import Asset
from finquant.returns import daily_returns, historical_mean_return


class Stock(Asset):
    """Class that contains information about a stock/fund.

    :param investmentinfo: Investment information for the stock as a ``pandas.DataFrame``.
    :param data: Historical price data for the stock as a ``pandas.DataFrame``.

    The ``Stock`` class extends the ``Asset`` class and represents a specific type of asset,
    namely a stock within a portfolio.
    It requires investment information and historical price data for the stock to initialize an instance.

    In addition to the attributes inherited from the ``Asset`` class, the ``Stock`` class provides
    a method to compute the beta parameter specific to stocks in a portfolio when compared to
    the market index.

    """


    def __init__(self, investmentinfo: pd.DataFrame, data: pd.Series) -> None:
        """
        :Input:
         :investmentinfo: ``pandas.DataFrame`` of investment information
         :data: ``pandas.Series`` of stock price
        """
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        super().__init__(data, self.name, asset_type="Stock")
        # beta parameter of stock (CAPM)
        self.beta = None

    def comp_beta(self, market_daily_returns: pd.Series) -> float:
        """Compute and return the Beta parameter of the stock.

        :Input:
         :market_daily_returns: ``pd.Series``, daily returns of the market

        :Output:
         :beta: ``float``, the Beta parameter of the stock
        """
        cov_mat = np.cov(
            self.comp_daily_returns(),
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
        string += f"\n{self.asset_type}: {self.name}"
        string += f"\nExpected Return: {self.expected_return:0.3f}"
        string += f"\nVolatility: {self.volatility:0.3f}"
        string += f"\nSkewness: {self.skew:0.5f}"
        string += f"\nKurtosis: {self.kurtosis:0.5f}"
        if self.beta is not None:
            string += f"\n{self.asset_type} Beta: {self.beta:0.3f}"
        string += "\nInformation:"
        string += "\n" + str(self.investmentinfo.to_frame().transpose())
        string += "\n" + "-" * 50
        print(string)