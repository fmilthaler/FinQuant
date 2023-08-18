"""
This module provides a public class ``Stock`` that represents a single stock or fund.
Instances of this class are used within the ``Portfolio`` class (provided in ``finquant.portfolio``).

The ``Stock`` class is designed to hold and calculate quantities related to a single stock or fund.
To initialize an instance of ``Stock``, it requires the following information:

    - ``investmentinfo``: Information about the stock or fund provided as a ``pandas.DataFrame``.
        The required column labels are ``Name`` and ``Allocation`` for the stock/fund name and allocation,
        respectively. However, the DataFrame can contain more information beyond these columns,
        such as year, strategy, currency (CCY), etc.

    - ``data``: Historical price data for the stock or fund provided as a ``pandas.Series``.
        The data must contain ``<stock_name> - Adj. Close``, which represents the closing price used to
        compute the return on investment.

The ``Stock`` class computes various quantities related to the stock or fund, such as expected return,
volatility, skewness, and kurtosis. It also provides functionality to calculate the beta parameter
using the CAPM (Capital Asset Pricing Model) and the R squared value of the stock .

The ``Stock`` class inherits from the ``Asset`` class in ``finquant.asset``, which provides common
functionality and attributes for financial assets.

"""

from typing import Optional

import numpy as np
import pandas as pd

from finquant.asset import Asset
from finquant.data_types import FLOAT
from finquant.type_utilities import type_validation

from sklearn.metrics import r2_score


class Stock(Asset):
    """Class that contains information about a stock/fund.

    :param investmentinfo: Investment information of a stock.
    :param data: Historical price data of a stock.

    The ``Stock`` class extends the ``Asset`` class and represents a specific type of asset,
    namely a stock within a portfolio.
    It requires investment information and historical price data for the stock to initialize an instance.

    In addition to the attributes inherited from the ``Asset`` class, the ``Stock`` class provides
    a method to compute the beta parameter and one to compute the R squared coefficient
    specific to stocks in a portfolio when compared to the market index.

    """

    # Attributes:
    investmentinfo: pd.DataFrame
    beta: Optional[FLOAT]
    rsquared: Optional[FLOAT]

    def __init__(self, investmentinfo: pd.DataFrame, data: pd.Series) -> None:
        """
        :param investmentinfo: Investment information of a stock.
        :param data: Historical price data of a stock.
        """
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        super().__init__(data, self.name, asset_type="Stock")
        # beta parameter of stock (CAPM)
        self.beta = None
        # R squared coefficient of stock
        self.rsquared = None

    def comp_beta(self, market_daily_returns: pd.Series) -> FLOAT:
        """Computes and returns the Beta parameter of the stock.

        :param market_daily_returns: Daily returns of the market index.

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: Beta parameter of the stock
        """
        # Type validations:
        type_validation(market_daily_returns=market_daily_returns)
        cov_mat = np.cov(
            self.comp_daily_returns(),
            market_daily_returns.to_frame()[market_daily_returns.name],
        )

        beta = float(cov_mat[0, 1] / cov_mat[1, 1])
        self.beta = beta
        return beta

    def comp_rsquared(self, market_daily_returns: pd.Series) -> FLOAT:
        """Computes and returns the R squared coefficient of the stock.

        :param market_daily_returns: Daily returns of the market index.

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: R squared coefficient of the stock
        """
        # Type validations:
        type_validation(market_daily_returns=market_daily_returns)

        rsquared = r2_score(
            market_daily_returns.to_frame()[market_daily_returns.name],
            self.comp_daily_returns(),
        )
        self.rsquared = rsquared
        return rsquared

    def properties(self) -> None:
        """Nicely prints out the properties of the stock: Expected Return,
        Volatility, Beta (optional), R squared (optional), Skewness, Kurtosis as well as the ``Allocation``
        (and other information provided in investmentinfo.)
        """
        # nicely printing out information and quantities of the stock
        string = "-" * 50
        string += f"\n{self.asset_type}: {self.name}"
        string += f"\nExpected Return: {self.expected_return:0.3f}"
        string += f"\nVolatility: {self.volatility:0.3f}"
        if self.beta is not None:
            string += f"\n{self.asset_type} Beta: {self.beta:0.3f}"
        if self.rsquared is not None:
            string += f"\n{self.asset_type} R squared: {self.rsquared:0.3f}"
        string += f"\nSkewness: {self.skew:0.5f}"
        string += f"\nKurtosis: {self.kurtosis:0.5f}"
        string += "\nInformation:"
        string += "\n" + str(self.investmentinfo.to_frame().transpose())
        string += "\n" + "-" * 50
        print(string)
