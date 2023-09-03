"""This module is the **core** of `FinQuant`. It provides

- a public class ``Portfolio`` that holds and calculates quantities of a financial
  portfolio, which is a collection of ``Stock`` instances (the ``Stock`` class is
  provided in ``finquant.stock`` and is a child class of ``Asset`` defined in
  ``finquant.asset``).
- a public function ``build_portfolio()`` that automatically constructs and returns
  an instance of ``Portfolio`` and instances of ``Stock``. 
  The relevant stock data is either retrieved through `quandl`/`yfinance` or provided by the user as a
  ``pandas.DataFrame`` (after loading it manually from disk/reading from file).
  For an example on how to use it, please read the corresponding docstring,
  or have a look at the examples in the sub-directory ``example``.

The class ``Portfolio`` is designed to easily manage your financial portfolio, 
and makes the most common quantitative calculations, such as:

- cumulative returns of the portfolio's stocks
- daily returns of the portfolio's stocks (daily percentage change),
- daily log returns of the portfolio's stocks,
- Expected (annualised) Return,
- Volatility,
- Downside Risk,
- Value at Risk,
- Sharpe Ratio,
- Sortino Ratio,
- Treynor Ratio (optional),
- Beta parameter (optional),
- R squared coefficient (optional),
- skewness of the portfolio's stocks,
- Kurtosis of the portfolio's stocks,
- the portfolio's covariance matrix.

Furthermore, the constructed portfolio can be optimised for

- minimum Volatility,
- maximum Sharpe Ratio
- minimum Volatility for a given Expected Return
- maximum Sharpe Ratio for a given target Volatility

by either performing a numerical computation to solve a minimisation problem,
or by performing a Monte Carlo simulation of `n` trials.
The former should be the preferred method for reasons of computational effort
and accuracy. The latter is only included for the sake of completeness.

Finally, functions are implemented to generate the following plots:

- Monte Carlo run to find optimal portfolio(s)
- Efficient Frontier
- Portfolio with the minimum Volatility based a numerical optimisation
- Portfolio with the maximum Sharpe Ratio based on a numerical optimisation
- Portfolio with the minimum Volatility for a given Expected Return based
  on a numerical optimisation
- Portfolio with the maximum Sharpe Ratio for a given target Volatility
  based on a numerical optimisation
- Individual stocks of the portfolio (Expected Return over Volatility)
"""
# supress some pylint complaints for this module only
# pylint: disable=C0302,R0904,,R0912,W0212

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from finquant.data_types import (
    ARRAY_OR_LIST,
    ELEMENT_TYPE,
    FLOAT,
    INT,
    LIST_DICT_KEYS,
    NUMERIC,
    STRING_OR_DATETIME,
)
from finquant.efficient_frontier import EfficientFrontier
from finquant.exceptions import (
    InvalidDateFormatError,
    QuandlError,
    QuandlLimitError,
    YFinanceError,
)
from finquant.market import Market
from finquant.monte_carlo import MonteCarloOpt
from finquant.quants import (
    downside_risk,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    value_at_risk,
    weighted_mean,
    weighted_std,
)
from finquant.returns import (
    cumulative_returns,
    daily_log_returns,
    daily_returns,
    historical_mean_return,
)
from finquant.stock import Stock
from finquant.type_utilities import type_validation


class Portfolio:
    """Object that contains information about an investment portfolio.
    To initialise the object, it does not require any input.
    To fill the portfolio with investment information, the
    function ``add_stock(stock)`` should be used, in which ``stock`` is
    an object of ``Stock``.
    """

    # Attributes:
    portfolio: pd.DataFrame
    stocks: Dict[str, Stock]
    data: pd.DataFrame
    expected_return: FLOAT
    volatility: FLOAT
    downside_risk: FLOAT
    var: FLOAT
    sharpe: FLOAT
    sortino: FLOAT
    treynor: Optional[FLOAT]
    skew: pd.Series
    kurtosis: pd.Series
    __totalinvestment: NUMERIC
    __var_confidence_level: FLOAT
    __risk_free_rate: FLOAT
    __freq: INT
    ef: Optional[EfficientFrontier]
    mc: Optional[MonteCarloOpt]
    __market_index: Optional[Market]
    beta_stocks: pd.DataFrame
    beta: Optional[FLOAT]
    rsquared_stocks: pd.DataFrame
    rsquared: Optional[FLOAT]

    def __init__(self) -> None:
        """Initiates ``Portfolio``."""
        # initilisating instance variables
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.data = pd.DataFrame()
        self.__var_confidence_level = 0.95
        self.__risk_free_rate = 0.005
        self.__freq = 252
        # instance variables for Efficient Frontier and Monte Carlo optimisations
        self.ef = None
        self.mc = None
        # instance variable for Market class
        self.__market_index = None
        # Treynor Ratio of the portfolio
        self.treynor = None
        # dataframe containing beta parameters of stocks
        self.beta_stocks = pd.DataFrame(index=["beta"])
        self.beta = None
        # dataframe containing rsquared coefficients of stocks
        self.rsquared_stocks = pd.DataFrame(index=["rsquared"])
        self.rsquared = None

    @property
    def totalinvestment(self) -> NUMERIC:
        return self.__totalinvestment

    @totalinvestment.setter
    def totalinvestment(self, val: NUMERIC) -> None:
        if val is not None:
            # treat "None" as initialisation
            if not isinstance(val, (float, int, np.floating, np.integer)):
                raise ValueError("Total investment must be a float or integer.")
            if val <= 0:
                raise ValueError(
                    "The money to be invested in the portfolio must be > 0."
                )
            self.__totalinvestment = val

    @property
    def freq(self) -> INT:
        return self.__freq

    @freq.setter
    def freq(self, val: INT) -> None:
        if not isinstance(val, (int, np.integer)):
            raise ValueError("Time window/frequency must be an integer.")
        if val <= 0:
            raise ValueError("freq must be > 0.")
        self.__freq = val
        # now that this changed, update other quantities
        self._update()

    @property
    def risk_free_rate(self) -> FLOAT:
        return self.__risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, val: FLOAT) -> None:
        if not isinstance(val, (float, np.floating)):
            raise ValueError("Risk free rate must be a float.")
        self.__risk_free_rate = val
        # now that this changed, update other quantities
        self._update()

    @property
    def market_index(self) -> Optional[Market]:
        return self.__market_index

    @market_index.setter
    def market_index(self, index: Market) -> None:
        """Set the market index to the portfolio.

        :param index: An object of the ``Market`` class.
        """
        self.__market_index = index

    @property
    def var_confidence_level(self) -> FLOAT:
        return self.__var_confidence_level

    @var_confidence_level.setter
    def var_confidence_level(self, val: FLOAT) -> None:
        if not isinstance(val, (float, np.floating)):
            raise ValueError("confidence level is expected to be a float.")
        if val >= 1 or val <= 0:
            raise ValueError("confidence level is expected to be between 0 and 1.")
        self.__var_confidence_level = val
        # now that this changed, update VaR
        self._update()

    def add_stock(self, stock: Stock, defer_update: bool = False) -> None:
        """Adds a stock of type ``Stock`` to the portfolio. Each time ``add_stock``
        is called, the following instance variables are updated:

        - ``portfolio``: ``pandas.DataFrame``, adds a column with information from ``stock``
        - ``stocks``: ``dictionary``, adds an entry for ``stock``
        - ``data``: ``pandas.DataFrame``, adds a column of stock prices from ``stock``

        Also, if argument ``defer_update`` is ``True``,
        the following instance variables are (re-)computed:

        - ``expected_return``: Expected Return of the portfolio
        - ``volatility``: Volatility of the portfolio
        - ``downside_risk``: Downside Risk
        - ``var``: Value at Risk of the portfolio
        - ``sharpe``: Sharpe Ratio of the portfolio
        - ``sortino``: Sortino Ratio of the portfolio
        - ``skew``: Skewness of the portfolio's stocks
        - ``kurtosis``: Kurtosis of the portfolio's stocks

        If argument ``defer_update`` is ``True`` and ``__market_index`` is not ``None``,
        the following instance variables are (re-)computed as well:

        - ``beta``: Beta parameter of the portfolio
        - ``rsquared``: R squared coefficient of the portfolio
        - ``treynor``: Treynor Ratio of the portfolio

        :param stock: An instance of the class ``Stock``.
        :param defer_update: bool, if True instance variables are not (re-)computed at the end of this method.
        """
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name: stock})
        # adding information of stock to the portfolio
        self.portfolio = pd.concat(
            [self.portfolio, stock.investmentinfo.to_frame().T], ignore_index=True
        )
        # setting an appropriate name for the portfolio
        self.portfolio.name = "Allocation of stocks"
        # also add stock data of stock to the dataframe
        self._add_stock_data(stock)

        if not defer_update:
            # update quantities of portfolio
            self._update()

    def _add_stock_data(self, stock: Stock) -> None:
        # insert given data into portfolio stocks dataframe:
        self.data.insert(
            loc=len(self.data.columns), column=stock.name, value=stock.data
        )
        # set index correctly
        self.data.set_index(stock.data.index.values, inplace=True)
        # set index name:
        self.data.index.rename("Date", inplace=True)

        if self.market_index is not None:
            # compute beta parameter of stock
            beta_stock = stock.comp_beta(self.market_index.daily_returns)
            # add beta of stock to portfolio's betas dataframe
            self.beta_stocks[stock.name] = [beta_stock]
            # compute R squared coefficient of stock
            rsquared_stock = stock.comp_rsquared(self.market_index.daily_returns)
            # add rsquared of stock to portfolio's R squared dataframe
            self.rsquared_stocks[stock.name] = [rsquared_stock]

    def _update(self) -> None:
        # sanity check (only update values if none of the below is empty):
        if not (self.portfolio.empty or not self.stocks or self.data.empty):
            self.totalinvestment = self.portfolio.Allocation.sum()
            self.expected_return = self.comp_expected_return(freq=self.freq)
            self.volatility = self.comp_volatility(freq=self.freq)
            self.downside_risk = self.comp_downside_risk(freq=self.freq)
            self.var = self.comp_var()
            self.sharpe = self.comp_sharpe()
            self.sortino = self.comp_sortino()
            self.skew = self._comp_skew()
            self.kurtosis = self._comp_kurtosis()
            if self.market_index is not None:
                self.beta = self.comp_beta()
                self.rsquared = self.comp_rsquared()
                self.treynor = self.comp_treynor()

    def get_stock(self, name: str) -> Stock:
        """Returns the instance of ``Stock`` with name ``name``.

        :param name: String of the name of the stock that is returned. Must match
             one of the labels in the dictionary ``pf.stocks``.

        :return: Instance of ``Stock`` taken from the portfolio.
        """
        return self.stocks[name]

    def comp_cumulative_returns(self) -> pd.DataFrame:
        """Computes the cumulative returns of all stocks in the portfolio.
        See ``finquant.returns.cumulative_returns``.

        :return: Cumulative returns of given stock prices.
        """
        return cumulative_returns(self.data)

    def comp_daily_returns(self) -> pd.DataFrame:
        """Computes the daily returns (percentage change) of all
        stocks in the portfolio. See ``finquant.returns.daily_returns``.

        :return: Daily percentage change of Returns of given stock prices.
        """
        return daily_returns(self.data)

    def comp_daily_log_returns(self) -> pd.DataFrame:
        """Computes the daily log returns of all stocks in the portfolio.
        See ``finquant.returns.daily_log_returns``.

        :return: Daily log Returns of given stock prices.
        """
        return daily_log_returns(self.data)

    def comp_mean_returns(self, freq: INT = 252) -> pd.Series:
        """Computes the mean returns based on historical stock price data.
        See ``finquant.returns.historical_mean_return``.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`

        :return: Historical mean Returns.
        """
        # Type validations:
        type_validation(freq=freq)
        return historical_mean_return(self.data, freq=freq)

    def comp_stock_volatility(self, freq: INT = 252) -> pd.Series:
        """Computes the Volatilities of all the stocks individually

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`

        :return: Individual volatilities of all stocks in the portfolio.
        """
        # Type validations:
        type_validation(freq=freq)
        return self.comp_daily_returns().std() * np.sqrt(freq)

    def comp_weights(self) -> pd.Series:
        """Computes and returns a ``pandas.Series`` of the weights/allocation
        of the stocks of the portfolio.

        :return: A Series with weights/allocation of all stocks within the portfolio.
        """
        # computes the weights of the stocks in the given portfolio
        # in respect of the total investment
        return (self.portfolio["Allocation"] / self.totalinvestment).astype(np.float64)

    def comp_expected_return(self, freq: INT = 252) -> FLOAT:
        """Computes the Expected Return of the portfolio.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: Expected Return of the portfolio.
        """
        # Type validations:
        type_validation(freq=freq)
        pf_return_means: pd.Series = historical_mean_return(self.data, freq=freq)
        weights: pd.Series = self.comp_weights()
        expected_return: FLOAT = weighted_mean(pf_return_means.values, weights)
        self.expected_return = expected_return
        return expected_return

    def comp_volatility(self, freq: INT = 252) -> FLOAT:
        """Computes the Volatility of the given portfolio.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: The volatility of the portfolio.
        """
        # Type validations:
        type_validation(freq=freq)
        # computing the volatility of a portfolio
        volatility: FLOAT = weighted_std(
            self.comp_cov(), self.comp_weights()
        ) * np.sqrt(freq)
        self.volatility = volatility
        return volatility

    def comp_downside_risk(self, freq: INT = 252) -> FLOAT:
        """Computes the downside risk of the portfolio.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`

        :return: Downside risk of the portfolio.
        """
        downs_risk: FLOAT = downside_risk(
            self.data, self.comp_weights(), self.risk_free_rate
        ) * np.sqrt(freq)
        self.downside_risk = downs_risk
        return downs_risk

    def comp_cov(self) -> pd.DataFrame:
        """Compute and return a DataFrame of the covariance matrix
        of the portfolio.

        :return: Covariance matrix of the portfolio.
        """
        # get the covariance matrix of the mean returns of the portfolio
        returns = daily_returns(self.data)
        return returns.cov()

    def comp_sharpe(self) -> FLOAT:
        """Compute and return the Sharpe Ratio of the portfolio.

        :type freq: :py:data:`~.finquant.data_types.FLOAT`
        :return: The Sharpe Ratio of the portfolio.
        """
        # compute the Sharpe Ratio of the portfolio
        sharpe: FLOAT = sharpe_ratio(
            self.expected_return, self.volatility, self.risk_free_rate
        )
        self.sharpe = sharpe
        return sharpe

    def comp_var(self) -> FLOAT:
        """Compute and return the Value at Risk of the portfolio.

        :type freq: :py:data:`~.finquant.data_types.FLOAT`
        :return: The Value at Risk (VaR) of the portfolio.
        """
        # compute the Value at Risk of the portfolio
        var: FLOAT = value_at_risk(
            investment=self.totalinvestment,
            mu=self.expected_return,
            sigma=self.volatility,
            conf_level=self.var_confidence_level,
        )
        self.var = var
        return var

    def comp_beta(self) -> Optional[FLOAT]:
        """Compute and return the Beta parameter of the portfolio.

        :return: Beta parameter of the portfolio
        """

        # compute the Beta parameter of the portfolio
        weights: pd.Series = self.comp_weights()
        if weights.size == self.beta_stocks.size:
            beta: FLOAT = weighted_mean(
                self.beta_stocks.transpose()["beta"].values, weights
            )

            self.beta = beta
            return beta
        else:
            return None

    def comp_rsquared(self) -> Optional[FLOAT]:
        """Compute and return the R squared coefficient of the portfolio.

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: R squared coefficient of the portfolio
        """

        # compute the R squared coefficient of the portfolio
        weights: pd.Series = self.comp_weights()
        if weights.size == self.beta_stocks.size:
            rsquared: FLOAT = weighted_mean(
                self.rsquared_stocks.transpose()["rsquared"].values, weights
            )

            self.rsquared = rsquared
            return rsquared
        else:
            return None

    def comp_sortino(self) -> FLOAT:
        """Compute and return the Sortino Ratio of the portfolio

        :type freq: :py:data:`~.finquant.data_types.FLOAT`
        :return: The Sortino Ratio of the portfolio.
            May be ``NaN`` if the portoflio outperformed the risk free rate at every point
        """
        return sortino_ratio(
            self.expected_return, self.downside_risk, self.risk_free_rate
        )

    def comp_treynor(self) -> Optional[FLOAT]:
        """Compute and return the Treynor Ratio of the portfolio.

        :rtype: :py:data:`~.finquant.data_types.FLOAT`
        :return: The Treynor Ratio of the portfolio.
        """
        # compute the Treynor Ratio of the portfolio
        treynor: Optional[FLOAT] = treynor_ratio(
            self.expected_return, self.beta, self.risk_free_rate
        )
        self.treynor = treynor
        return treynor

    def _comp_skew(self) -> pd.Series:
        """Computes and returns the skewness of the stocks in the portfolio."""
        return self.data.skew()

    def _comp_kurtosis(self) -> pd.Series:
        """Computes and returns the Kurtosis of the stocks in the portfolio."""
        return self.data.kurt()

    # optimising the investments with the efficient frontier class
    def _get_ef(self) -> EfficientFrontier:
        """If self.ef does not exist, create and return an instance of
        finquant.efficient_frontier.EfficientFrontier, else, return the
        existing instance.
        """
        if self.ef is None:
            # create instance of EfficientFrontier
            self.ef = EfficientFrontier(
                self.comp_mean_returns(freq=1),
                self.comp_cov(),
                risk_free_rate=self.risk_free_rate,
                freq=self.freq,
            )
        return self.ef

    def ef_minimum_volatility(self, verbose: bool = False) -> pd.DataFrame:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.minimum_volatility``.

        Finds the portfolio with the minimum Volatility.

        :param verbose: Whether to print out properties or not, default: False

        :return: A DataFrame of weights/allocation of stocks within the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # perform optimisation
        opt_weights: pd.DataFrame = ef.minimum_volatility()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_maximum_sharpe_ratio(self, verbose: bool = False) -> pd.DataFrame:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.maximum_sharpe_ratio``.

        Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        :param verbose: Whether to print out properties or not, default: False

        :return: A DataFrame of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # perform optimisation
        opt_weights: pd.DataFrame = ef.maximum_sharpe_ratio()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_return(
        self, target: NUMERIC, verbose: bool = False
    ) -> pd.DataFrame:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_return``.

        Finds the portfolio with the minimum Volatility for a given target return.

        :param target: The target return of the optimised portfolio.
        :type target: :py:data:`~.finquant.data_types.NUMERIC`

        :param verbose: Whether to print out properties or not, default: False

        :return: A DataFrame of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # perform optimisation
        opt_weights: pd.DataFrame = ef.efficient_return(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_volatility(
        self, target: NUMERIC, verbose: bool = False
    ) -> pd.DataFrame:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_volatility``.

        Finds the portfolio with the maximum Sharpe Ratio for a given
        target Volatility.

        :param target: The target return of the optimised portfolio.
        :type target: :py:data:`~.finquant.data_types.NUMERIC`

        :param verbose: Whether to print out properties or not, default: False

        :return: A DataFrame of weights/allocation of stocks within
             the optimised portfolio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # perform optimisation
        opt_weights: pd.DataFrame = ef.efficient_volatility(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_frontier(
        self, targets: Optional[ARRAY_OR_LIST[FLOAT]] = None
    ) -> np.ndarray[np.float64, Any]:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.efficient_frontier``.

        Gets portfolios for a range of given target Returns.
        If no targets were provided, the algorithm will find the minimum
        and maximum Returns of the portfolio's individual stocks, and set
        the target range according to those values.
        Results in the Efficient Frontier.

        :param targets: A list/array: range of target returns, default: ``None``

        :return: Efficient Frontier as an array of (volatility, Return) values
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # perform optimisation
        efrontier: np.ndarray[np.float64, Any] = ef.efficient_frontier(targets)
        return efrontier

    def ef_plot_efrontier(self) -> None:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.plot_efrontier``.

        Plots the Efficient Frontier."""
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # plot efficient frontier
        ef.plot_efrontier()

    def ef_plot_optimal_portfolios(self) -> None:
        """Interface to
        ``finquant.efficient_frontier.EfficientFrontier.plot_optimal_portfolios``.

        Plots markers of the optimised portfolios for

        - minimum Volatility, and
        - maximum Sharpe Ratio.
        """
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef: EfficientFrontier = self._get_ef()
        # plot efficient frontier
        ef.plot_optimal_portfolios()

    # optimising the investments with the efficient frontier class
    def _get_mc(self, num_trials: int = 1000) -> MonteCarloOpt:
        """If self.mc does not exist, create and return an instance of
        finquant.monte_carlo.MonteCarloOpt, else, return the existing instance.
        """
        if self.mc is None:
            # create instance of MonteCarloOpt
            self.mc = MonteCarloOpt(
                self.comp_daily_returns(),
                num_trials=num_trials,
                risk_free_rate=self.risk_free_rate,
                freq=self.freq,
                initial_weights=self.comp_weights().values,
            )
        return self.mc

    # optimising the investments by performing a Monte Carlo run
    # based on volatility and sharpe ratio
    def mc_optimisation(
        self, num_trials: int = 1000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Interface to
        ``finquant.monte_carlo.MonteCarloOpt.optimisation``.

        Optimisation of the portfolio by performing a Monte Carlo
        simulation.

        :param num_trials: Number of portfolios to be computed, each with a random distribution
            of weights/allocation in each stock, default: 1000

        :return:
            :opt_w: DataFrame with optimised investment strategies for maximum
                Sharpe Ratio and minimum volatility.
            :opt_res: DataFrame with Expected Return, Volatility and Sharpe Ratio
                for portfolios with minimum Volatility and maximum Sharpe Ratio.
        """
        # dismiss previous instance of mc, as we are performing a new MC optimisation:
        self.mc = None
        # get instance of MonteCarloOpt
        mc: MonteCarloOpt = self._get_mc(num_trials)
        opt_weights: pd.DataFrame
        opt_results: pd.DataFrame
        opt_weights, opt_results = mc.optimisation()
        return opt_weights, opt_results

    def mc_plot_results(self) -> None:
        """Plots the results of the Monte Carlo run, with all of the randomly
        generated weights/portfolios, as well as markers for the portfolios with the
        minimum Volatility, and maximum Sharpe Ratio.
        """
        # get instance of MonteCarloOpt
        mc: MonteCarloOpt = self._get_mc()
        mc.plot_results()

    def mc_properties(self) -> None:
        """Calculates and prints out Expected annualised Return,
        Volatility and Sharpe Ratio of optimised portfolio.
        """
        # get instance of MonteCarloOpt
        mc: MonteCarloOpt = self._get_mc()
        mc.properties()

    def plot_stocks(self, freq: INT = 252) -> None:
        """Plots the Expected annual Returns over annual Volatility of
        the stocks of the portfolio.

        :param freq: Number of trading days in a year, default: 252
        :type freq: :py:data:`~.finquant.data_types.INT`
        """
        # Type validations:
        type_validation(freq=freq)
        # annual mean returns of all stocks
        stock_returns: pd.Series = self.comp_mean_returns(freq=freq)
        stock_volatility: pd.Series = self.comp_stock_volatility(freq=freq)
        # adding stocks of the portfolio to the plot
        # plot stocks individually:
        plt.scatter(stock_volatility, stock_returns, marker="o", s=100, label="Stocks")
        # adding text to stocks in plot:
        for idx, txt in enumerate(stock_returns.index):
            plt.annotate(
                txt,
                (stock_volatility[idx], stock_returns[idx]),
                xytext=(10, 0),
                textcoords="offset points",
                label=idx,
            )

    def properties(self) -> None:
        """
        Nicely prints out the properties of the portfolio:

            - Expected Return,
            - Volatility,
            - Downside Risk,
            - Value at Risk (VaR),
            - Confidence level of VaR,
            - Sharpe Ratio,
            - Sortino Ratio,
            - Treynor Ratio (optional),
            - Beta (optional),
            - R squared (optional),
            - skewness,
            - Kurtosis

        as well as the allocation of the stocks across the portfolio.

        :rtype: None
        """
        # nicely printing out information and quantities of the portfolio
        string: str = "-" * 70
        stocknames = self.portfolio.Name.values.tolist()
        string += f"\nStocks: {', '.join(stocknames)}"
        if self.market_index is not None:
            string += f"\nMarket Index: {self.market_index.name}"
        string += f"\nTime window/frequency: {self.freq}"
        string += f"\nRisk free rate: {self.risk_free_rate}"
        string += f"\nPortfolio Expected Return: {self.expected_return:0.3f}"
        string += f"\nPortfolio Volatility: {self.volatility:0.3f}"
        string += f"\nPortfolio Downside Risk: {self.downside_risk:0.3f}"
        string += f"\nPortfolio Value at Risk: {self.var:0.3f}"
        string += "\nConfidence level of Value at Risk: "
        string += f"{self.var_confidence_level * 100:0.2f} %"
        string += f"\nPortfolio Sharpe Ratio: {self.sharpe:0.3f}"
        string += f"\nPortfolio Sortino Ratio: {self.sortino:0.3f}"
        if self.treynor is not None:
            string += f"\nPortfolio Treynor Ratio: {self.treynor:0.3f}"
        if self.beta is not None:
            string += f"\nPortfolio Beta: {self.beta:0.3f}"
        if self.rsquared is not None:
            string += f"\nPortfolio R squared: {self.rsquared:0.3f}"
        string += "\n\nSkewness:"
        string += "\n" + str(self.skew.to_frame().transpose())
        string += "\n\nKurtosis:"
        string += "\n" + str(self.kurtosis.to_frame().transpose())
        string += "\n\nInformation:"
        string += "\n" + str(self.portfolio)
        string += "\n"
        string += "-" * 70
        print(string)

    def __str__(self) -> str:
        # print short description
        return "Contains information about a portfolio."


def _correct_quandl_request_stock_name(
    names: Union[str, ARRAY_OR_LIST[str]]
) -> List[str]:
    """If given input argument is of type string,
    this function converts it to a list, assuming the input argument
    is only one stock name.
    """
    # Type validations:
    type_validation(names=names)
    # make sure names is a list of names:
    names_list: List[str]
    if isinstance(names, str):
        names_list = list(names)
    elif isinstance(names, np.ndarray):
        names_list = names.tolist()
    else:
        names_list = names
    return names_list


def _quandl_request(
    names: List[str],
    start_date: Optional[STRING_OR_DATETIME] = None,
    end_date: Optional[STRING_OR_DATETIME] = None,
) -> pd.DataFrame:
    """This function performs a simple request from `quandl` and returns
    a DataFrame containing stock data.

    :param names: List of strings of stock names to be requested
    :param start_date: String/datetime of the start date of relevant stock data.
    :param end_date: String/datetime of the end date of relevant stock data.
    """
    try:
        import quandl  # pylint: disable=C0415
    except ImportError:
        print(
            "The following package is required:\n - `quandl`\n"
            + "Please make sure that it is installed."
        )
    # Type validations:
    type_validation(names=names, start_date=start_date, end_date=end_date)

    # get correct stock names that quandl.get can request,
    # e.g. "WIKI/GOOG" for Google
    reqnames: List[str] = _correct_quandl_request_stock_name(names)
    try:
        resp: pd.DataFrame = quandl.get(
            reqnames, start_date=start_date, end_date=end_date
        )
    except quandl.LimitExceededError as exc:
        errormsg = (
            "You exceeded Quandl's limit. Are you using your API key?\nQuandl Error: "
            + str(exc)
        )
        raise QuandlLimitError(errormsg) from exc
    except Exception as exc:
        errormsg = (
            "An error occurred while retrieving data from Quandl.\n"
            + "Make sure all the requested stock names/tickers are "
            + "supported by Quandl.\n"
            + "Quandl error: "
            + str(exc)
        )
        raise QuandlError(errormsg) from exc

    return resp


def _yfinance_request(
    names: List[str],
    start_date: Optional[STRING_OR_DATETIME] = None,
    end_date: Optional[STRING_OR_DATETIME] = None,
) -> pd.DataFrame:
    """This function performs a simple request from Yahoo Finance
    (using `yfinance`) and returns a DataFrame containing stock data.

    :param names: List of strings of stock names to be requested
    :param start_date: (optional) String/datetime of the start date of relevant stock data.
    :param end_date: (optional) String/datetime of the end date of relevant stock data.
    """
    try:
        import yfinance  # pylint: disable=C0415
    except ImportError:
        print(
            "The following package is required:\n - `yfinance`\n"
            + "Please make sure that it is installed."
        )
    # Type validations:
    type_validation(names=names, start_date=start_date, end_date=end_date)

    # yfinance does not exit safely if start/end date were not given correctly:
    # this step is not required for quandl as it handles this exception properly
    try:
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise InvalidDateFormatError(
            "Please provide valid values for <start_date> and <end_date> "
            "(either as datetime object or as String in the format '%Y-%m-%d')."
        ) from exc

    # unlike quandl, yfinance does not have a prefix in front of the ticker
    # thus we do not need to correct them
    try:
        resp: pd.DataFrame = yfinance.download(names, start=start_date, end=end_date)
        if not isinstance(resp.columns, pd.MultiIndex) and len(names) > 0:
            # for single stock must make the dataframe multiindex
            stock_tuples = [(col, names[0]) for col in list(resp.columns)]
            resp.columns = pd.MultiIndex.from_tuples(stock_tuples)
    except Exception as exc:
        errormsg: str = (
            "An error occurred while retrieving data from Yahoo Finance with `yfinance`.\n"
            + "yfinance error: "
            + str(exc)
        )
        raise YFinanceError(errormsg) from exc
    return resp


def _get_quandl_data_column_label(stock_name: str, data_label: str) -> str:
    """Given stock name and label of a data column, this function returns
    the string "<stock_name> - <data_label>" as it can be found in a
    ``pandas.DataFrame`` returned by `quandl`.
    """
    return stock_name + " - " + data_label


def _get_stocks_data_columns(
    data: pd.DataFrame, names: ARRAY_OR_LIST[str], cols: List[str]
) -> pd.DataFrame:
    """This function returns a subset of the given ``pandas.DataFrame`` data, which
    contains only the data columns as specified in the input cols.

    :param data: A DataFrame which contains quantities of the stocks listed in pf_allocation.
    :param names: A list of strings, containing the names of the stocks, e.g. 'Google'.
    :param cols: A list of strings of column labels of data to be extracted.
        Currently only one column per stock is supported.

    :return: A DataFrame which contains only the data columns of data as specified in cols.
    """
    # Type validations:
    type_validation(data=data, names=names, cols=cols)
    # get correct stock names that quandl get request
    reqnames: List[str] = _correct_quandl_request_stock_name(names)
    # get current column labels and replacement labels
    reqcolnames: List[str] = []
    colname: str
    # if dataframe is of type multiindex, also get first level colname
    firstlevel_colnames: List[str] = []
    for idx, name in enumerate(names):
        for col in cols:
            # differ between dataframe directly from quandl and
            # possibly previously processed dataframe, e.g.
            # read in from disk with slightly modified column labels
            # 1. if <stock_name> in column labels
            if name in data.columns:
                colname = name
            # 2. if "WIKI/<stock_name> - <col>" in column labels
            elif _get_quandl_data_column_label(reqnames[idx], col) in data.columns:
                colname = _get_quandl_data_column_label(reqnames[idx], col)
            # 3. if "<stock_name> - <col>" in column labels
            elif _get_quandl_data_column_label(name, col) in data.columns:
                colname = _get_quandl_data_column_label(name, col)
            # if column labels are of type multiindex, and the "Adj Close" is in
            # first-level labels, we assume the dataframe comes from yfinance:
            elif isinstance(data.columns, pd.MultiIndex):
                # alter col for yfinance, as it returns column labels without '.'
                col = col.replace(".", "")
                if col in data.columns:
                    if not col in firstlevel_colnames:
                        firstlevel_colnames.append(col)
                    if name in data[col].columns:
                        colname = name
                    else:  # error, it must find name on the second level of the column header
                        raise ValueError(
                            "Could not find column labels in the second level of MultiIndex pd.DataFrame"
                        )
            # else, error
            else:
                raise ValueError("Could not find column labels in the given dataframe.")
            # append the correct name to the list of correct names
            reqcolnames.append(colname)

    # if data comes from yfinance, it is a multiindex dataframe:
    if isinstance(data.columns, pd.MultiIndex):
        if len(firstlevel_colnames) != 1:
            raise ValueError(
                "Sorry, for now only one value/quantity per Stock is supported."
            )
        data = data[firstlevel_colnames[0]].loc[:, reqcolnames]
    else:
        # if it comes from quandl, it is not of type multiindex
        data = data.loc[:, reqcolnames]

    # if only one data column per stock exists, rename column labels
    # to the name of the corresponding stock
    newcolnames: Dict[str, str] = {}
    if len(cols) == 1:
        for idx, name in enumerate(names):
            newcolnames.update({_get_quandl_data_column_label(name, cols[0]): name})
        data.rename(columns=newcolnames, inplace=True)
    return data


def _build_portfolio_from_api(
    names: ARRAY_OR_LIST[str],
    pf_allocation: Optional[pd.DataFrame] = None,
    start_date: Optional[STRING_OR_DATETIME] = None,
    end_date: Optional[STRING_OR_DATETIME] = None,
    data_api: str = "quandl",
    market_index: Optional[str] = None,
) -> Portfolio:
    """Returns a portfolio based on input in form of a list of strings/names
    of stocks.

    :param names: A list of strings, containing the names of the stocks, e.g. 'GOOG' for Google.
    :param pf_allocation: (optional) A DataFrame with the required data column labels ``Name`` and
        ``Allocation`` of the stocks.

    :param start_date: (optional) String/datetime of the start date of relevant stock data.
    :param end_date: (optional) String/datetime of the end date of relevant stock data.
    :param data_api: (optional, default: 'quandl') A string which determines what API to use to obtain stock prices,
        if data is not provided by the user. Valid values:
         - ``quandl`` (Python package/API to `Quandl`)
         - ``yfinance`` (Python package formerly known as ``fix-yahoo-finance``)
    :param market_index: (optional, default: ``None``) A string which determines the market index
        to be used for the computation of the Trenor Ratio, beta parameter and the R squared of the portfolio.

    :return: Instance of Portfolio which contains all the information requested by the user.
    """
    # Type validations:
    type_validation(
        names=names,
        pf_allocation=pf_allocation,
        start_date=start_date,
        end_date=end_date,
        data_api=data_api,
        market_index=market_index,
    )

    # setting up variables:
    stock_data: pd.DataFrame
    # create empty dataframe for market data
    market_data: pd.DataFrame = pd.DataFrame()
    # request data from service:
    if data_api == "yfinance":
        stock_data = _yfinance_request(list(names), start_date, end_date)
        if market_index is not None:
            market_data = _yfinance_request([market_index], start_date, end_date)
    elif data_api == "quandl":
        stock_data = _quandl_request(list(names), start_date, end_date)
        if market_index is not None:
            # only generated if user explicitly requests market index with quandl
            raise Warning("Market index is not supported for quandl data.")
    else:
        raise ValueError(
            f"Error: value of data_api '{data_api}' is not supported. "
            + "Choose between 'yfinance' and 'quandl'."
        )
    # check pf_allocation:
    if pf_allocation is None:
        pf_allocation = _generate_pf_allocation(names=list(names))
    # build portfolio:
    pf: Portfolio = _build_portfolio_from_df(
        stock_data, pf_allocation, market_data=market_data
    )
    return pf


def _stocknames_in_data_columns(names: ARRAY_OR_LIST[str], df: pd.DataFrame) -> bool:
    """Returns True if at least one element of names was found as a column
    label in the dataframe df.
    """
    return any((name in label for name in names for label in df.columns))


def _get_index_adj_clos_pr(data: pd.DataFrame) -> pd.Series:
    """This function returns a subset of the given ``pandas.DataFrame`` data, which
    contains only the data columns corresponding to Adjusted Closing Price.

    :param data: A DataFrame which contains financial data.

    :return: A Series which contains only the data column of data corresponding to Adjusted Closing Price.
    """
    return data["Adj Close"].squeeze().astype(np.float64)


def _generate_pf_allocation(
    names: Optional[List[str]] = None, data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Takes column names of provided ``pandas.DataFrame`` ``data``, and generates a
    ``pandas.DataFrame`` with columns ``Name`` and ``Allocation`` which contain the
    names found in input ``data`` and 1.0/len(data.columns) respectively.

    :param data: A DataFrame which contains prices of the stocks.

    :return: A DataFrame with columns ``Name`` and ``Allocation``, which contain the names
        and weights of the stocks.
    """
    # checking input arguments
    if names is not None and data is not None or names is None and data is None:
        raise ValueError("Pass one of the two: 'names' or 'data'.")
    # Type validations:
    type_validation(names=names, data=data)

    # defining new variable stock_names to circumvent the issue of "names" being Optional
    stock_names: List[str]

    # if data is given:
    if data is not None:
        # this case is more complex, as we need to check for column labels in
        # data
        stock_names = data.columns.tolist()
        # potential error message
        errormsg: str = (
            "'data' pandas.DataFrame contains conflicting column labels."
            + "\nMultiple columns with a substring of\n {}\n"
            + "were found. You have two options:"
            + "\n 1. call 'build_portfolio' and pass a pandas.DataFrame "
            + "'pf_allocation' that contains the weights/allocation of stocks "
            + "within your portfolio. 'build_portfolio' will then extract the "
            + "columns from 'data' that match the values of the column 'Name' in "
            + "the pandas.DataFrame 'pf_allocation'."
            + "\n 2. call 'build_portfolio' and pass a pandas.DataFrame 'data' "
            + "that does not have conflicting column labels, e.g. 'GOOG' and "
            + "'GOOG - Adj. Close' are considered conflicting column headers."
        )
        # sanity check: split stock_names at '-' and take the leading part of the
        # split string, and check if this occurs in any of the other stock_names.
        # if so, we treat this as a duplication, and ask the user to provide
        # a DataFrame with one data column per stock.
        splitnames: List[str] = [name.split("-")[0].strip() for name in stock_names]
        for idx, splitname in enumerate(splitnames):
            reducedlist: List[str] = [
                elt for num, elt in enumerate(splitnames) if num != idx
            ]
            if splitname in reducedlist:
                errormsg = errormsg.format(str(splitname))
                raise ValueError(errormsg)
    elif names is not None:
        # if names is given, we use names as stock_names:
        stock_names = names
    # no else needed, this is already covered at the beginning of the function
    # compute equal weights
    weights = [1.0 / float(len(stock_names)) for _ in range(len(stock_names))]
    return pd.DataFrame({"Allocation": weights, "Name": stock_names})


def _build_portfolio_from_df(
    data: pd.DataFrame,
    pf_allocation: pd.DataFrame = None,
    data_columns: Optional[List[str]] = None,
    market_data: pd.DataFrame = None,
) -> Portfolio:
    """Returns a portfolio based on input in form of ``pandas.DataFrame``.

    :param data: A DataFrame which contains prices of the stocks listed in pf_allocation.
    :param pf_allocation: (optional) A DataFrame with the required data column
         labels ``Name`` and ``Allocation`` of the stocks. If not given, it is
         automatically generated with an equal weights for all stocks
         in the resulting portfolio, default: ``None``.
    :param data_columns: (optional) A list of strings of data column labels
         to be extracted and returned (default: ``["Adj. Close"]``).
    :param market_data: (optional) A DataFrame which contains data of the
         market index (default: ``None``).

    :return: Instance of Portfolio which contains all the information requested by the user.
    """
    # if pf_allocation is None, automatically generate it
    if pf_allocation is None:
        pf_allocation = _generate_pf_allocation(data=data)
    if data_columns is None:
        data_columns = ["Adj. Close"]
    # Enforcing types for pf_allocation:
    pf_allocation = pf_allocation.astype({"Allocation": np.float64, "Name": str})
    # make sure stock names are in data dataframe
    if not _stocknames_in_data_columns(pf_allocation.Name.values, data):
        raise ValueError(
            "Error: None of the provided stock names were"
            + "found in the provided dataframe."
        )
    # Enforce np.float64 for data columns:
    data = data.astype(np.float64)
    # extract only "Adjusted Close" price column from DataFrame:
    # in quandl: "Adj. Close"; in yfinance: "Adj Close"
    data = _get_stocks_data_columns(data, pf_allocation.Name.values, data_columns)

    # building portfolio:
    pf: Portfolio = Portfolio()
    if market_data is not None and not market_data.empty:
        # extract only "Adjusted Close" price column from market data
        market_data = _get_index_adj_clos_pr(market_data)
        # set market index of portfolio
        pf.market_index = Market(data=market_data)
    for idx in range(len(pf_allocation)):
        # get name of stock
        name: str = pf_allocation.iloc[idx].Name
        # extract data column of said stock
        stock_data: pd.Series = data.loc[:, [name]].copy(deep=True).squeeze()
        # create Stock instance and add it to portfolio,
        # and defer updating portfolio attributes until all stocks are added
        pf.add_stock(
            Stock(investmentinfo=pf_allocation.iloc[idx], data=stock_data),
            defer_update=True,
        )
    # update the portfolio
    pf._update()
    return pf


def _all_list_ele_in_other(
    l_1: LIST_DICT_KEYS[ELEMENT_TYPE], l_2: LIST_DICT_KEYS[ELEMENT_TYPE]
) -> bool:
    """Returns True if all elements of list l1 are found in list l2."""
    return all(ele in l_2 for ele in l_1)


def _any_list_ele_in_other(
    l_1: LIST_DICT_KEYS[ELEMENT_TYPE], l_2: LIST_DICT_KEYS[ELEMENT_TYPE]
) -> bool:
    """Returns True if any element of list l1 is found in list l2."""
    return any(ele in l_2 for ele in l_1)


def _list_complement(
    set_a: LIST_DICT_KEYS[ELEMENT_TYPE], set_b: LIST_DICT_KEYS[ELEMENT_TYPE]
) -> List[ELEMENT_TYPE]:
    """Returns the relative complement of A in B (also denoted as A\\B)"""
    return list(set(set_b) - set(set_a))


def build_portfolio(**kwargs: Dict[str, Any]) -> Portfolio:
    """This function builds and returns an instance of ``Portfolio``
    given a set of input arguments.

    :param pf_allocation: (optional) A DataFrame with the required data column
         labels ``Name`` and ``Allocation`` of the stocks. If not given, it is
         automatically generated with an equal weights for all stocks
         in the resulting portfolio.
    :param names: (optional) A List of strings, containing the names
         of the stocks, e.g. "GOOG" for Google.
    :param start_date: (optional) string/datetime start date of stock data to be
         requested through `quandl`/`yfinance` (default: ``None``).
    :param end_date: (optional) string/datetime end date of stock data to be
         requested through `quandl`/`yfinance` (default: ``None``).
    :param data: (optional) A DataFrame which contains quantities of
         the stocks listed in ``pf_allocation``.
    :param data_api: (optional) A string (default: ``quandl``) which determines how to obtain
        stock prices, if data is not provided by the user. Valid values:

         - ``quandl`` (Python package/API to `Quandl`)
         - ``yfinance`` (Python package formerly known as ``fix-yahoo-finance``)

    :param market_index: (optional) A string (default: ``None``) which determines the
         market index to be used for the computation of the Treynor ratio, beta parameter 
         and the R squared coefficient of the portflio.

    :return: Instance of ``Portfolio`` which contains all the information requested by the user.

    .. note:: Only the following combinations of inputs are allowed:

     - ``names``, ``pf_allocation`` (optional), ``start_date`` (optional), ``end_date`` (optional),\
        ``data_api`` (optional), ``market_index`` (optional)
     - ``data``, ``pf_allocation`` (optional)

     The two different ways this function can be used are useful for:

     1. building a portfolio by pulling data from `quandl`/`yfinance`,
     2. building a portfolio by providing stock data which was obtained otherwise,
        e.g. from data files.

     If used in an unsupported way, the function (or subsequently called function)
     raises appropriate Exceptions with useful information what went wrong.
    """
    docstring_msg: str = (
        "Please read through the docstring, "
        "'build_portfolio.__doc__' and/or have a look at the "
        "examples in `examples/`."
    )
    input_error: str = (
        "You passed an unsupported argument to "
        "build_portfolio. The following arguments are not "
        "supported:"
        "\n {}\nOnly the following arguments are allowed:\n "
        "{}\n" + docstring_msg
    )
    input_comb_error: str = (
        "Error: None of the input arguments {} are allowed "
        "in combination with {}.\n" + docstring_msg
    )

    # list of all valid optional input arguments
    all_input_args: List[str] = [
        "pf_allocation",
        "names",
        "start_date",
        "end_date",
        "data",
        "data_api",
        "market_index",
    ]

    # check if no input argument was passed
    if not kwargs:
        raise ValueError(
            "Error:\nbuild_portfolio() requires input " + "arguments.\n" + docstring_msg
        )
    # check for valid input arguments
    if not _all_list_ele_in_other(kwargs.keys(), all_input_args):
        unsupported_input: List[str] = _list_complement(all_input_args, kwargs.keys())
        raise ValueError(
            "Error:\n" + input_error.format(unsupported_input, all_input_args)
        )

    # create an empty portfolio
    pf: Portfolio = Portfolio()

    # 1. pf_allocation, names, start_date, end_date, data_api, market_index
    allowed_mandatory_args: List[str] = ["names"]
    allowed_input_args: List[str] = [
        "names",
        "pf_allocation",
        "start_date",
        "end_date",
        "data_api",
        "market_index",
    ]
    complement_input_args: List[str] = _list_complement(
        allowed_input_args, all_input_args
    )

    if _all_list_ele_in_other(allowed_mandatory_args, kwargs.keys()):
        # check that no input argument conflict arises:
        if _any_list_ele_in_other(complement_input_args, kwargs.keys()):
            raise ValueError(
                input_comb_error.format(complement_input_args, allowed_mandatory_args)
            )

        # Extract given/potential arguments from kwargs:
        names = cast(List[str], list(kwargs.get("names", [])))
        pf_allocation = kwargs.get("pf_allocation", None)
        start_date = cast(Optional[STRING_OR_DATETIME], kwargs.get("start_date", None))
        end_date = cast(Optional[STRING_OR_DATETIME], kwargs.get("end_date", None))
        data_api = cast(str, kwargs.get("data_api", "quandl"))
        market_index = cast(Optional[str], kwargs.get("market_index", None))

        # get portfolio:
        pf = _build_portfolio_from_api(
            names=names,
            pf_allocation=pf_allocation,
            start_date=start_date,
            end_date=end_date,
            data_api=data_api,
            market_index=market_index,
        )

    # 2. pf_allocation, data
    allowed_mandatory_args = ["data"]
    allowed_input_args = ["data", "pf_allocation"]
    complement_input_args = _list_complement(allowed_input_args, all_input_args)
    if _all_list_ele_in_other(allowed_mandatory_args, kwargs.keys()):
        # check that no input argument conflict arises:
        if _any_list_ele_in_other(complement_input_args, kwargs.keys()):
            raise ValueError(
                input_comb_error.format(complement_input_args, allowed_mandatory_args)
            )

        # Extract given/potential arguments from kwargs:
        data = kwargs.get("data", pd.DataFrame())
        pf_allocation = kwargs.get("pf_allocation", None)

        # get portfolio:
        pf = _build_portfolio_from_df(data=data, pf_allocation=pf_allocation)

    # final check
    # pylint: disable=R0916
    if (
        pf.portfolio.empty
        or pf.data.empty
        or not pf.stocks
        or not hasattr(pf, "expected_return")
        or not hasattr(pf, "volatility")
        or not hasattr(pf, "downside_risk")
        or not hasattr(pf, "var")
        or not hasattr(pf, "sharpe")
        or not hasattr(pf, "sortino")
        or pf.skew.empty
        or pf.kurtosis.empty
    ):
        raise ValueError(
            "Should not get here. Something went wrong while "
            + "creating an instance of Portfolio. "
            + docstring_msg
        )

    return pf
