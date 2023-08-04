"""The module provides a class ``MonteCarlo`` which is an implementation of the
Monte Carlo method and a class ``MonteCarloOpt`` which allows the user to perform a
Monte Carlo run to find optimised financial portfolios, given an initial portfolio.
"""


from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from finquant.data_types import FLOAT, INT
from finquant.quants import annualised_portfolio_quantities
from finquant.type_utilities import type_validation


class MonteCarlo:
    """An object to perform a Monte Carlo run/simulation."""

    # Attributes:
    num_trials: int

    def __init__(self, num_trials: int = 1000):
        """
        :param num_trials: Number of iterations of the Monte Carlo run/simulation, default: 1000
        """
        self.num_trials = num_trials

    def run(
        self, fun: Callable[..., Any], **kwargs: Dict[str, Any]
    ) -> np.ndarray[np.float64, Any]:
        """
        :param fun: Function to call at each iteration of the Monte Carlo run.

        :param kwargs: (optional) Additional arguments that are passed to ``fun``.

        :result: Array of quantities returned from ``fun`` at each iteration.
        """
        # Type validations:
        type_validation(fun=fun)
        result = []
        for _ in range(self.num_trials):
            res = fun(**kwargs)
            result.append(res)
        return np.asarray(result, dtype=np.ndarray)


class MonteCarloOpt(MonteCarlo):
    """An object to perform a Monte Carlo run/simulation for finding
    optimised financial portfolios.

    Inherits from `MonteCarlo`.
    """

    # Attributes:
    returns: pd.DataFrame
    risk_free_rate: FLOAT
    freq: INT
    initial_weights: Optional[np.ndarray[np.float64, Any]]

    def __init__(
        self,
        returns: pd.DataFrame,
        num_trials: int = 1000,
        risk_free_rate: FLOAT = 0.005,
        freq: INT = 252,
        initial_weights: Optional[np.ndarray[np.float64, Any]] = None,
    ) -> None:
        """
        :param returns: DataFrame of returns of stocks
             Note: If applicable, the given returns should be computed with the same risk free rate
             and time window/frequency (arguments ``risk_free_rate`` and ``freq`` as passed in here.
        :param num_trials: Number of portfolios to be computed,
            each with a random distribution of weights/allocation in each stock, default: 1000
        :param risk_free_rate: Risk free rate as required for the Sharpe Ratio, default: 0.005
        :param freq: Number of trading days in a year, default: 252
        :param initial_weights: Weights of initial/given portfolio, only used to plot a marker for the
             initial portfolio in the optimisation plot, default: ``None``
        """
        # Type validations:
        type_validation(
            returns_df=returns,
            num_trials=num_trials,
            risk_free_rate=risk_free_rate,
            freq=freq,
            initial_weights=initial_weights,
        )
        self.returns = returns
        self.num_trials = num_trials
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        self.initial_weights: np.ndarray[float, Any] = initial_weights
        # initiate super class
        super().__init__(num_trials=self.num_trials)
        # setting additional variables
        self.num_stocks = len(self.returns.columns)
        self.return_means = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        # setting up variables for results
        self.df_weights = None
        self.df_results = None
        self.opt_weights = None
        self.opt_results = None

    def _random_weights(
        self,
    ) -> Tuple[np.ndarray[np.float64, Any], np.ndarray[np.float64, Any]]:
        """Computes random weights for the stocks of a portfolio and the
        corresponding Expected Return, Volatility and Sharpe Ratio.

        :result: Tuple of (weights (array), and array of expected return, volatility, sharpe ratio)
        """
        # select random weights for portfolio
        weights: np.ndarray[np.float64, Any] = np.array(
            np.random.random(self.num_stocks), dtype=np.float64
        )
        # rebalance weights
        weights = weights / np.sum(weights)
        # compute portfolio return and volatility
        portfolio_values: np.ndarray[np.float64, Any] = np.array(
            annualised_portfolio_quantities(
                weights,
                self.return_means,
                self.cov_matrix,
                self.risk_free_rate,
                self.freq,
            ),
            dtype=np.float64,
        )
        return (weights, portfolio_values)

    def _random_portfolios(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs a Monte Carlo run and gets a list of random portfolios
        and their corresponding quantities (Expected Return, Volatility,
        Sharpe Ratio).

        :return:
            :df_weights: DataFrame, holds the weights for each randomly generated portfolio
            :df_results: DataFrame, holds Expected Annualised Return, Volatility and
                Sharpe Ratio of each randomly generated portfolio
        """
        # run Monte Carlo to get random weights and corresponding quantities
        res = self.run(self._random_weights)
        # convert to pandas.DataFrame:
        weights_columns = list(self.returns.columns)
        result_columns = ["Expected Return", "Volatility", "Sharpe Ratio"]
        df_weights = pd.DataFrame(
            data=res[:, 0].tolist(), columns=weights_columns
        ).astype(np.float64)
        df_results = pd.DataFrame(
            data=res[:, 1].tolist(), columns=result_columns
        ).astype(np.float64)
        return (df_weights, df_results)

    def optimisation(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimisation of the portfolio by performing a Monte Carlo
        simulation.

        :return:
            :opt_w: DataFrame with optimised investment strategies for maximum
                Sharpe Ratio and minimum volatility.
            :opt_res: DataFrame with Expected Return, Volatility and Sharpe Ratio
                for portfolios with minimum Volatility and maximum Sharpe Ratio.
        """
        # perform Monte Carlo run and get weights and results
        df_weights, df_results = self._random_portfolios()
        # finding portfolios with the minimum volatility and maximum
        # Sharpe ratio
        index_min_volatility = df_results["Volatility"].idxmin()
        index_max_sharpe = df_results["Sharpe Ratio"].idxmax()
        # storing optimal results to DataFrames
        opt_w: pd.DataFrame = pd.DataFrame(
            [df_weights.iloc[index_min_volatility], df_weights.iloc[index_max_sharpe]],
            index=["Min Volatility", "Max Sharpe Ratio"],
        )
        opt_res: pd.DataFrame = pd.DataFrame(
            [df_results.iloc[index_min_volatility], df_results.iloc[index_max_sharpe]],
            index=["Min Volatility", "Max Sharpe Ratio"],
        )
        # setting instance variables:
        self.df_weights = df_weights.astype(np.float64)
        self.df_results = df_results.astype(np.float64)
        self.opt_weights = opt_w.astype(np.float64)
        self.opt_results = opt_res.astype(np.float64)
        return opt_w, opt_res

    def plot_results(self) -> None:
        """Plots the results of the Monte Carlo run, with all of the
        randomly generated weights/portfolios, as well as markers
        for the portfolios with the minimum Volatility and maximum
        Sharpe Ratio.
        """
        if (
            self.df_results is None
            or self.df_weights is None
            or self.opt_weights is None
            or self.opt_results is None
        ):
            raise ValueError(
                "Error: Cannot plot, run the Monte Carlo " + "optimisation first."
            )
        # create scatter plot coloured by Sharpe Ratio
        plt.scatter(
            self.df_results["Volatility"],
            self.df_results["Expected Return"],
            c=self.df_results["Sharpe Ratio"],
            cmap="RdYlBu",
            s=10,
            label=None,
        )
        cbar = plt.colorbar()
        # mark in green the minimum volatility
        plt.scatter(
            self.opt_results.loc["Min Volatility"]["Volatility"],
            self.opt_results.loc["Min Volatility"]["Expected Return"],
            marker="^",
            color="g",
            s=100,
            label="min Volatility",
        )
        # mark in red the highest sharpe ratio
        plt.scatter(
            self.opt_results.loc["Max Sharpe Ratio"]["Volatility"],
            self.opt_results.loc["Max Sharpe Ratio"]["Expected Return"],
            marker="^",
            color="r",
            s=100,
            label="max Sharpe Ratio",
        )
        # also set marker for initial portfolio, if weights were given
        if self.initial_weights is not None:
            # computed expected return and volatility of initial portfolio
            initial_values = annualised_portfolio_quantities(
                self.initial_weights,
                self.return_means,
                self.cov_matrix,
                self.risk_free_rate,
                self.freq,
            )
            initial_return = initial_values[0]
            initial_volatility = initial_values[1]
            plt.scatter(
                initial_volatility,
                initial_return,
                marker="^",
                color="k",
                s=100,
                label="Initial Portfolio",
            )
        plt.title(
            "Monte Carlo simulation to optimise the portfolio based "
            + "on the Efficient Frontier"
        )
        plt.xlabel("Volatility [period=" + str(self.freq) + "]")
        plt.ylabel("Expected Return [period=" + str(self.freq) + "]")
        cbar.ax.set_ylabel("Sharpe Ratio [period=" + str(self.freq) + "]", rotation=90)
        plt.legend()

    def properties(self) -> None:
        """Prints out the properties of the Monte Carlo optimisation."""
        if self.opt_weights is None or self.opt_results is None:
            print(
                "Error: Optimal weights and/or results are not computed. Please perform a Monte Carlo run first."
            )
        else:
            # print out results
            opt_vals = ["Min Volatility", "Max Sharpe Ratio"]
            string = ""
            for val in opt_vals:
                string += "-" * 70
                string += f"\nOptimised portfolio for {val.replace('Min', 'Minimum').replace('Max', 'Maximum')}"
                string += f"\n\nTime period: {self.freq} days"
                string += f"\nExpected return: {self.opt_results.loc[val]['Expected Return']:0.3f}"
                string += (
                    f"\nVolatility: {self.opt_results.loc[val]['Volatility']:0.3f}"
                )
                string += (
                    f"\nSharpe Ratio: {self.opt_results.loc[val]['Sharpe Ratio']:0.3f}"
                )
                string += "\n\nOptimal weights:"
                string += "\n" + str(
                    self.opt_weights.loc[val]
                    .to_frame()
                    .transpose()
                    .rename(index={val: "Allocation"})
                )
                string += "\n"
            string += "-" * 70
            print(string)
