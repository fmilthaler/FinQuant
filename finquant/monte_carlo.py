"""The module provides a class ``MonteCarlo`` which is an implementation of the
Monte Carlo method and a class ``MonteCarloOpt`` which allows the user to perform a
Monte Carlo run to find optimised financial portfolios, given an intial portfolio.
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from finquant.quants import annualised_portfolio_quantities


class MonteCarlo(object):
    """An object to perform a Monte Carlo run/simulation."""

    def __init__(self, num_trials=1000):
        """
        :Input:
         :num_trials: ``int`` (default: ``1000``), number of iterations of the
                 Monte Carlo run/simulation.
        """
        self.num_trials = num_trials

    def run(self, fun, **kwargs):
        """
        :Input:
         :fun: Function to call at each iteration of the Monte Carlo run.
         :kwargs: (optional) Additional arguments that are passed to `fun`.

        :Output:
         :result: List of quantities returned from `fun` at each iteration.
        """
        result = []
        for i in range(self.num_trials):
            res = fun(**kwargs)
            result.append(res)
        return np.asarray(result)


class MonteCarloOpt(MonteCarlo):
    """An object to perform a Monte Carlo run/simulation for finding
    optimised financial portfolios.

    Inherits from `MonteCarlo`.
    """

    def __init__(
        self,
        returns,
        num_trials=1000,
        risk_free_rate=0.005,
        freq=252,
        initial_weights=None,
    ):
        """
        :Input:
         :returns: A ``pandas.DataFrame`` which contains the returns of stocks.
             Note: If applicable, the given returns should be computed with the
             same risk free rate and time window/frequency (arguments
             ``risk_free_rate`` and ``freq`` as passed down here.
         :num_trials: ``int`` (default: ``1000``), number of portfolios to be
             computed, each with a random distribution of weights/allocation
             in each stock
         :risk_free_rate: ``float`` (default: ``0.005``), the risk free rate as
             required for the Sharpe Ratio
         :freq: ``int`` (default: ``252``), number of trading days, default
             value corresponds to trading days in a year
         :initial_weights: ``list``/``numpy.ndarray`` (default: ``None``), weights of
             initial/given portfolio, only used to plot a marker for the
             initial portfolio in the optimisation plot.

        :Output:
         :opt: ``pandas.DataFrame`` with optimised investment strategies for maximum
             Sharpe Ratio and minimum volatility.
        """
        if initial_weights is not None and not isinstance(initial_weights, np.ndarray):
            raise ValueError(
                "If given, optional argument 'initial_weights' "
                + "must be of type numpy.ndarray"
            )
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("returns is expected to be a pandas.DataFrame")
        if not isinstance(num_trials, int):
            raise ValueError("num_trials is expected to be an integer")
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate is expected to be an integer or float.")
        if not isinstance(freq, int):
            raise ValueError("freq is expected to be an integer.")
        self.returns = returns
        self.num_trials = num_trials
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        self.initial_weights = initial_weights
        # initiate super class
        super(MonteCarloOpt, self).__init__(num_trials=self.num_trials)
        # setting additional variables
        self.num_stocks = len(self.returns.columns)
        self.return_means = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        # setting up variables for results
        self.df_weights = None
        self.df_results = None
        self.opt_weights = None
        self.opt_results = None

    def _random_weights(self):
        """Computes random weights for the stocks of a portfolio and the
        corresponding Expected Return, Volatility and Sharpe Ratio.

        :Output:
         :(weights, quantities): Tuple of weights (np.ndarray) and a
             list of [expected return, volatility, sharpe ratio].
        """
        # select random weights for portfolio
        w = np.array(np.random.random(self.num_stocks))
        # rebalance weights
        w = w / np.sum(w)
        # compute portfolio return and volatility
        portfolio_values = annualised_portfolio_quantities(
            w, self.return_means, self.cov_matrix, self.risk_free_rate, self.freq
        )
        return (w, np.array(portfolio_values))

    def _random_portfolios(self):
        """Performs a Monte Carlo run and gets a list of random portfolios
        and their corresponding quantities (Expected Return, Volatility,
        Sharpe Ratio). Returns ``pandas.DataFrame`` of weights and results.

        :Output:
         :df_weights: ``pandas.DataFrame``, holds the weights for each randomly
             generated portfolio
         :df_results: ``pandas.DataFrame``, holds Expected Annualised Return,
             Volatility and Sharpe Ratio of each randomly generated portfolio
        """
        # run Monte Carlo to get random weights and corresponding quantities
        res = self.run(self._random_weights)
        # convert to pandas.DataFrame:
        weights_columns = list(self.returns.columns)
        result_columns = ["Expected Return", "Volatility", "Sharpe Ratio"]
        df_weights = pd.DataFrame(data=res[:, 0].tolist(), columns=weights_columns)
        df_results = pd.DataFrame(data=res[:, 1].tolist(), columns=result_columns)
        return (df_weights, df_results)

    def optimisation(self):
        """Optimisation of the portfolio by performing a Monte Carlo
        simulation.

        :Output:
         :opt_w: ``pandas.DataFrame`` with optimised investment strategies for maximum
             Sharpe Ratio and minimum volatility.
         :opt_res: ``pandas.DataFrame`` with Expected Return, Volatility and Sharpe Ratio
             for portfolios with minimum Volatility and maximum Sharpe Ratio.
        """
        # perform Monte Carlo run and get weights and results
        df_weights, df_results = self._random_portfolios()
        # finding portfolios with the minimum volatility and maximum
        # Sharpe ratio
        index_min_volatility = df_results["Volatility"].idxmin()
        index_max_sharpe = df_results["Sharpe Ratio"].idxmax()
        # storing optimal results to DataFrames
        opt_w = pd.DataFrame(
            [df_weights.iloc[index_min_volatility], df_weights.iloc[index_max_sharpe]],
            index=["Min Volatility", "Max Sharpe Ratio"],
        )
        opt_res = pd.DataFrame(
            [df_results.iloc[index_min_volatility], df_results.iloc[index_max_sharpe]],
            index=["Min Volatility", "Max Sharpe Ratio"],
        )
        # setting instance variables:
        self.df_weights = df_weights
        self.df_results = df_results
        self.opt_weights = opt_w
        self.opt_results = opt_res
        return opt_w, opt_res

    def plot_results(self):
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
            raise Exception(
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

    def properties(self):
        """Prints out the properties of the Monte Carlo optimisation."""
        # print out results
        opt_vals = ["Min Volatility", "Max Sharpe Ratio"]
        string = ""
        for val in opt_vals:
            string += "-" * 70
            string += "\nOptimised portfolio for {}".format(
                val.replace("Min", "Minimum").replace("Max", "Maximum")
            )
            string += "\n\nTime period: {} days".format(self.freq)
            string += "\nExpected return: {0:0.3f}".format(
                self.opt_results.loc[val]["Expected Return"]
            )
            string += "\nVolatility: {:0.3f}".format(
                self.opt_results.loc[val]["Volatility"]
            )
            string += "\nSharpe Ratio: {:0.3f}".format(
                self.opt_results.loc[val]["Sharpe Ratio"]
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
