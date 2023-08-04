"""
This module provides a public class ``Asset`` that represents a generic financial asset.
It serves as the parent class for specialized asset classes like ``Stock`` and ``Market``
in the finquant library.

An asset is characterized by its historical price data, from which various quantities
such as expected return, volatility, skewness, and kurtosis can be computed. The ``Asset``
class provides common functionality and attributes that are shared among different types
of assets.

The specialized asset classes, ``Stock`` and ``Market``, inherit from the ``Asset`` class
and add specific functionality tailored to their respective types.

"""

import numpy as np
import pandas as pd

from finquant.data_types import FLOAT, INT
from finquant.returns import daily_returns, historical_mean_return


class Asset:
    """
    Parent class representing a generic financial asset.

    :param data: Historical price data of the asset.
    :param name: Name of the asset.
    :param asset_type: Type of the asset (e.g., "Stock" or "Market index").

    The ``Asset`` class provides common functionality and attributes for financial assets.
    It represents a generic asset and serves as the parent class for specialized asset classes.

    Attributes:
        - ``data`` (``pandas.Series``): Historical price data of the asset.
        - ``name`` (``str``): Name of the asset.
        - ``asset_type`` (``str``): Type of the asset (e.g., "Stock" or "Market").
        - ``expected_return`` (``float``): Expected return of the asset.
        - ``volatility`` (``float``): Volatility of the asset.
        - ``skew`` (``float``): Skewness of the asset.
        - ``kurtosis`` (``float``): Kurtosis of the asset.

    The ``Asset`` class provides methods to compute various quantities such as expected return,
    volatility, skewness, and kurtosis based on the historical price data of the asset.

    """

    # Attributes:
    data: pd.Series
    name: str
    asset_type: str
    expected_return: pd.Series
    volatility: FLOAT
    skew: FLOAT
    kurtosis: FLOAT

    def __init__(
        self, data: pd.Series, name: str, asset_type: str = "Market index"
    ) -> None:
        """
        :param data: Historical price data of the asset.
        :param name: Name of the asset
        :param asset_type: Type of the asset (e.g., "Stock" or "Market index"), default: "Market index"
        """
        self.data = data.astype(np.float64)
        self.name = name
        # compute expected return and volatility of asset
        self.expected_return = self.comp_expected_return()
        self.volatility = self.comp_volatility()
        self.skew = self._comp_skew()
        self.kurtosis = self._comp_kurtosis()
        # type of asset
        self.asset_type = asset_type

    # functions to compute quantities
    def comp_daily_returns(self) -> pd.Series:
        """Computes the daily returns (percentage change) of the asset.
        See ``finquant.returns.daily_returns``.
        """
        return daily_returns(self.data)

    def comp_expected_return(self, freq: INT = 252) -> pd.Series:
        """Computes the Expected Return of the asset.
        See ``finquant.returns.historical_mean_return``.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.type_definitions.INT`
        """
        return historical_mean_return(self.data, freq=freq)

    def comp_volatility(self, freq: INT = 252) -> FLOAT:
        """Computes the Volatility of the asset.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.type_definitions.INT`

        :rtype: :py:data:`~.finquant.type_definitions.FLOAT`
        """
        volatility: FLOAT = self.comp_daily_returns().std() * np.sqrt(freq)
        return volatility

    def _comp_skew(self) -> FLOAT:
        """Computes and returns the skewness of the asset."""
        skew: FLOAT = self.data.skew()
        return skew

    def _comp_kurtosis(self) -> FLOAT:
        """Computes and returns the kurtosis of the asset."""
        kurtosis: FLOAT = self.data.kurt()
        return kurtosis

    def properties(self) -> None:
        """Nicely prints out the properties of the asset,
        with customized messages based on the asset type.
        """
        # nicely printing out information and quantities of the asset
        string = "-" * 50
        string += f"\n{self.asset_type}: {self.name}"
        string += f"\nExpected Return: {self.expected_return:0.3f}"
        string += f"\nVolatility: {self.volatility:0.3f}"
        string += f"\nSkewness: {self.skew:0.5f}"
        string += f"\nKurtosis: {self.kurtosis:0.5f}"
        string += "\n" + "-" * 50
        print(string)

    def __str__(self) -> str:
        # print short description
        string = f"Contains information about {self.asset_type}: {self.name}."
        return string
