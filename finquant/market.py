"""
This module provides a public class ``Market`` that represents a market index.
It serves as a specialized asset class within the finquant library.

A market index represents the performance of a specific market or a segment of the market,
such as the S&P 500 or NASDAQ. The ``Market`` class is designed to hold and calculate quantities
related to a market index, such as expected return, volatility, skewness, and kurtosis.

The ``Market`` class extends the ``Asset`` class from ``finquant.asset`` and inherits its
common functionality and attributes for financial assets. It provides additional methods
and attributes specific to market indices.

"""


import pandas as pd

from finquant.asset import Asset
from finquant.returns import daily_returns


class Market(Asset):
    """
    Class representing a market index.

    :param data: Historical price data of the market index.

    The ``Market`` class extends the ``Asset`` class and represents a specific type of asset,
    specifically a market index.
    It requires historical price data for the market index to initialize an instance.

    """

    def __init__(self, data: pd.Series) -> None:
        """
        :param data: Historical price data of the market index.
        """
        super().__init__(data, name=data.name, asset_type="Market index")
        self.daily_returns = self.comp_daily_returns()

    # functions to compute quantities
    def comp_daily_returns(self) -> pd.Series:
        """Computes the daily returns (percentage change) of the market index.
        See ``finquant.returns.daily_returns``.
        """
        return daily_returns(self.data)
