"""The module facilitates a class `EfficientFrontier` that can be used to
optimise a portfolio by minimising a cost/objective function.
"""


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco

import finquant.minimise_fun as min_fun
from finquant.data_types import ARRAY_OR_DATAFRAME, ARRAY_OR_LIST, FLOAT, INT, NUMERIC
from finquant.quants import annualised_portfolio_quantities


class EfficientFrontier:
    """An object designed to perform optimisations based on minimising a cost/objective function.
    It can find parameters for portfolios with

    - minimum volatility
    - maximum Sharpe ratio
    - minimum volatility for a given target return
    - maximum Sharpe ratio for a given target volatility

    It also provides functions to compute the Efficient Frontier between a range
    of Returns, plot the Efficient Frontier, plot the optimal portfolios
    (minimum Volatility and maximum Sharpe Ratio).
    """

    # Attributes:
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    risk_free_rate: FLOAT
    freq: INT
    method: str
    names: List[str]
    num_stocks: int
    last_optimisation: str
    bounds: Tuple[Tuple[int, int], ...]
    x_0: np.ndarray[np.float64, Any]
    constraints: Dict[str, Union[str, Callable[[Any], FLOAT]]]
    weights: np.ndarray[np.float64, Any]
    df_weights: pd.DataFrame
    efrontier: np.ndarray[np.float64, Any]

    def __init__(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: FLOAT = 0.005,
        freq: INT = 252,
        method: str = "SLSQP",
    ):
        """
        :Input:
         :mean_returns: ``pandas.Series``, individual expected returns for all
             stocks in the portfolio
         :cov_matrix: ``pandas.DataFrame``, covariance matrix of returns
         :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate
         :freq: ``int`` (default= ``252``), number of trading days, default
             value corresponds to trading days in a year
         :method: ``string`` (default= ``"SLSQP"``), type of solver method to use,
             must be one of:

             - 'Nelder-Mead'
             - 'Powell'
             - 'CG'
             - 'BFGS'
             - 'Newton-CG'
             - 'L-BFGS-B'
             - 'TNC'
             - 'COBYLA'
             - 'SLSQP'
             - 'trust-constr'
             - 'dogleg'
             - 'trust-ncg'
             - 'trust-exact'
             - 'trust-krylov'

             all of which are officially supported by scipy.optimize.minimize
        """
        if not isinstance(mean_returns, pd.Series):
            raise ValueError("mean_returns is expected to be a pandas.Series.")
        if not isinstance(cov_matrix, pd.DataFrame):
            raise ValueError("cov_matrix is expected to be a pandas.DataFrame")
        supported_methods = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        if not isinstance(risk_free_rate, (np.floating, float)):
            raise ValueError("risk_free_rate is expected to be a float.")
        if not isinstance(method, str):
            raise ValueError("method is expected to be a string.")
        if method not in supported_methods:
            raise ValueError("method is not supported by scipy.optimize.minimize.")

        # instance variables
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.freq = freq
        self.method = method
        self.names = list(mean_returns.index)
        self.num_stocks = len(self.names)
        self.last_optimisation = ""

        # set numerical parameters
        bound = (0, 1)
        self.bounds = tuple(bound for _ in range(self.num_stocks))
        self.x_0 = np.array(self.num_stocks * [1.0 / self.num_stocks], dtype=np.float64)
        self.constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        # placeholder for optimised values/weights
        self.weights = np.empty(0, dtype=np.float64)
        self.df_weights = pd.DataFrame()
        self.efrontier = np.empty((0, 2), dtype=np.float64)

    def minimum_volatility(self, save_weights: bool = True) -> ARRAY_OR_DATAFRAME[FLOAT]:
        """Finds the portfolio with the minimum volatility.

        :Input:
         :save_weights: ``boolean`` (default= ``True``), for internal use only.
             Whether to save the optimised weights in the instance variable
             ``weights`` (and ``df_weights``). Useful for the case of computing
             the efficient frontier after doing an optimisation, else the optimal
             weights would be overwritten by the efficient frontier computations.
             Best to ignore this argument.

        :Output:
         :df_weights:
           - if "save_weights" is True: a ``pandas.DataFrame`` of weights/allocation
             of stocks within the optimised portfolio.
         :weights:
           - if "save_weights" is False: a ``numpy.ndarray`` of weights/allocation
             of stocks within the optimised portfolio.
        """
        if not isinstance(save_weights, bool):
            raise ValueError("save_weights is expected to be a boolean.")

        args = (self.mean_returns.values, self.cov_matrix.values)

        # Optimization
        result = sco.minimize(
            min_fun.portfolio_volatility,
            args=args,
            x0=self.x_0,
            method=self.method,
            bounds=self.bounds,
            constraints=self.constraints,
        )

        # Set the last optimization
        self.last_optimisation = "Minimum Volatility"

        # Set optimal weights
        if save_weights:
            self.weights = result["x"]
            self.df_weights = self._dataframe_weights(self.weights)
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result["x"]

    def maximum_sharpe_ratio(self, save_weights: bool = True) -> ARRAY_OR_DATAFRAME[FLOAT]:
        """Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        :Input:
         :save_weights: ``boolean`` (default= ``True``), for internal use only.
             Whether to save the optimised weights in the instance variable
             ``weights`` (and ``df_weights``). Useful for the case of computing
             the efficient frontier after doing an optimisation, else the optimal
             weights would be overwritten by the efficient frontier computations.
             Best to ignore this argument.

        :Output:
         :df_weights:
           - if "save_weights" is True: a ``pandas.DataFrame`` of weights/allocation
             of stocks within the optimised portfolio.
         :weights:
           - if "save_weights" is False: a ``numpy.ndarray`` of weights/allocation
             of stocks within the optimised portfolio.
        """
        if not isinstance(save_weights, bool):
            raise ValueError("save_weights is expected to be a boolean.")
        args = (self.mean_returns.values, self.cov_matrix.values, self.risk_free_rate)
        # optimisation
        result = sco.minimize(
            min_fun.negative_sharpe_ratio,
            args=args,
            x0=self.x_0,
            method=self.method,
            bounds=self.bounds,
            constraints=self.constraints,
        )
        # if successful, set self.last_optimisation
        self.last_optimisation = "Maximum Sharpe Ratio"
        # set optimal weights
        if save_weights:
            self.weights = result["x"]
            self.df_weights = self._dataframe_weights(self.weights)
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result["x"]

    def efficient_return(
        self, target: NUMERIC, save_weights: bool = True
    ) -> ARRAY_OR_DATAFRAME[FLOAT]:
        """Finds the portfolio with the minimum volatility for a given target
        return.

        :Input:
         :target: ``float``, the target return of the optimised portfolio.
         :save_weights: ``boolean`` (default= ``True``), for internal use only.
             Whether to save the optimised weights in the instance variable
             ``weights`` (and ``df_weights``). Useful for the case of computing
             the efficient frontier after doing an optimisation, else the optimal
             weights would be overwritten by the efficient frontier computations.
             Best to ignore this argument.

        :Output:
         :df_weights:
           - if "save_weights" is True: a ``pandas.DataFrame`` of weights/allocation
             of stocks within the optimised portfolio.
         :weights:
           - if "save_weights" is False: a ``numpy.ndarray`` of weights/allocation
             of stocks within the optimised portfolio.
        """
        if not isinstance(target, (int, float)):
            raise ValueError("target is expected to be an integer or float.")
        if not isinstance(save_weights, bool):
            raise ValueError("save_weights is expected to be a boolean.")
        args = (self.mean_returns.values, self.cov_matrix.values)
        # here we have an additional constraint:
        constraints = (
            self.constraints,
            {
                "type": "eq",
                "fun": lambda x: min_fun.portfolio_return(
                    x, self.mean_returns, self.cov_matrix
                )
                - target,
            },
        )
        # optimisation
        result = sco.minimize(
            min_fun.portfolio_volatility,
            args=args,
            x0=self.x_0,
            method=self.method,
            bounds=self.bounds,
            constraints=constraints,
        )
        # if successful, set self.last_optimisation
        self.last_optimisation = "Efficient Return"
        # set optimal weights
        if save_weights:
            self.weights = result["x"]
            self.df_weights = self._dataframe_weights(self.weights)
            return self.df_weights
        else:
            # not setting instance variables, and returning array instead
            # of pandas.DataFrame
            return result["x"]

    def efficient_volatility(self, target: NUMERIC) -> pd.DataFrame:
        """Finds the portfolio with the maximum Sharpe ratio for a given
        target volatility.

        :Input:
         :target: ``float``, the target volatility of the optimised portfolio.

        :Output:
         :df_weights: a ``pandas.DataFrame`` of weights/allocation of stocks within
             the optimised portfolio.
        """
        if not isinstance(target, (int, float)):
            raise ValueError("target is expected to be an integer or float.")
        args = (self.mean_returns.values, self.cov_matrix.values, self.risk_free_rate)
        # here we have an additional constraint:
        constraints = (
            self.constraints,
            {
                "type": "eq",
                "fun": lambda x: min_fun.portfolio_volatility(
                    x, self.mean_returns, self.cov_matrix
                )
                - target,
            },
        )
        # optimisation
        result = sco.minimize(
            min_fun.negative_sharpe_ratio,
            args=args,
            x0=self.x_0,
            method=self.method,
            bounds=self.bounds,
            constraints=constraints,
        )
        # if successful, set self.last_optimisation
        self.last_optimisation = "Efficient Volatility"
        # set optimal weights
        self.weights = result["x"]
        self.df_weights = self._dataframe_weights(self.weights)
        return self.df_weights

    def efficient_frontier(
        self, targets: Optional[ARRAY_OR_LIST[FLOAT]] = None
    ) -> np.ndarray[np.float64, Any]:
        """Gets portfolios for a range of given target returns.
        If no targets were provided, the algorithm will find the minimum
        and maximum returns of the portfolio's individual stocks, and set
        the target range according to those values.
        Results in the Efficient Frontier.

        :Input:
         :targets: ``list``/``numpy.ndarray`` (default= ``None``) of ``floats``,
             range of target returns.

        :Output:
         :efrontier: ``numpy.ndarray`` of (volatility, return) values
        """
        if targets is not None and not isinstance(targets, (list, np.ndarray)):
            raise ValueError("targets is expected to be a list or numpy.ndarray")
        if targets is None:
            # set range of target returns from the individual expected
            # returns of the stocks in the portfolio.
            min_return = self.mean_returns.min() * self.freq
            max_return = self.mean_returns.max() * self.freq
            targets = np.linspace(round(min_return, 3), round(max_return, 3), 100)
        # compute the efficient frontier
        efrontier = []
        for target in targets:
            weights = self.efficient_return(target, save_weights=False)
            efrontier.append(
                [
                    annualised_portfolio_quantities(
                        weights, self.mean_returns, self.cov_matrix, freq=self.freq
                    )[1],
                    target,
                ]
            )
        self.efrontier: np.ndarray[np.float64, Any] = np.array(
            efrontier, dtype=np.float64
        )
        if self.efrontier.size == 0 or self.efrontier.ndim != 2:
            raise ValueError("Error: Efficient frontier could not be computed.")
        return self.efrontier

    def plot_efrontier(self) -> None:
        """Plots the Efficient Frontier."""
        if self.efrontier.size == 0:
            # compute efficient frontier first
            self.efficient_frontier()
        plt.plot(
            self.efrontier[:, 0],
            self.efrontier[:, 1],
            linestyle="-.",
            color="black",
            lw=2,
            label="Efficient Frontier",
        )
        plt.title("Efficient Frontier")
        plt.xlabel("Volatility")
        plt.ylabel("Expected Return")
        plt.legend()

    def plot_optimal_portfolios(self) -> None:
        """Plots markers of the optimised portfolios for

        - minimum Volatility, and
        - maximum Sharpe Ratio.
        """
        # compute optimal portfolios
        min_vol_weights = self.minimum_volatility(save_weights=False)
        max_sharpe_weights = self.maximum_sharpe_ratio(save_weights=False)
        # compute return and volatility for each portfolio
        min_vol_vals = list(
            annualised_portfolio_quantities(
                min_vol_weights, self.mean_returns, self.cov_matrix, freq=self.freq
            )
        )[0:2]
        min_vol_vals.reverse()
        max_sharpe_vals = list(
            annualised_portfolio_quantities(
                max_sharpe_weights, self.mean_returns, self.cov_matrix, freq=self.freq
            )
        )[0:2]
        max_sharpe_vals.reverse()
        plt.scatter(
            min_vol_vals[0],
            min_vol_vals[1],
            marker="X",
            color="g",
            s=150,
            label="EF min Volatility",
        )
        plt.scatter(
            max_sharpe_vals[0],
            max_sharpe_vals[1],
            marker="X",
            color="r",
            s=150,
            label="EF max Sharpe Ratio",
        )
        plt.legend()

    def _dataframe_weights(
        self, weights: Optional[np.ndarray[np.float64, Any]]
    ) -> pd.DataFrame:
        """Generates and returns a ``pandas.DataFrame`` from given
        array weights.

        :Input:
         :weights: ``numpy.ndarray``, weights of the stock of the portfolio

        :Output:
         :weights: ``pandas.DataFrame`` with the weights/allocation of stocks
        """
        if not isinstance(weights, np.ndarray):
            raise ValueError("weights is expected to be a numpy.ndarray")
        return pd.DataFrame(weights, index=self.names, columns=["Allocation"]).astype(
            np.float64
        )

    def properties(self, verbose: bool = False) -> Tuple[NUMERIC, FLOAT, FLOAT]:
        """Calculates and prints out Expected annualised Return,
        Volatility and Sharpe Ratio of optimised portfolio.

        :Input:
         :verbose: ``boolean`` (default= ``False``), whether to print out properties or not
        """
        if not isinstance(verbose, bool):
            raise ValueError("verbose is expected to be a boolean.")
        if self.weights.size == 0:
            raise ValueError(
                "Error: weights are empty. Please perform an optimisation first."
            )
        expected_return, volatility, sharpe = annualised_portfolio_quantities(
            self.weights,
            self.mean_returns,
            self.cov_matrix,
            risk_free_rate=self.risk_free_rate,
            freq=self.freq,
        )
        if verbose:
            string = "-" * 70
            string += f"\nOptimised portfolio for {self.last_optimisation}"
            string += f"\n\nTime window/frequency: {self.freq}"
            string += f"\nRisk free rate: {self.risk_free_rate}"
            string += f"\nExpected annual Return: {expected_return:.3f}"
            string += f"\nAnnual Volatility: {volatility:.3f}"
            string += f"\nSharpe Ratio: {sharpe:.3f}"
            string += "\n\nOptimal weights:"
            string += f"\n{str(self.df_weights.transpose())}"
            string += "\n"
            string += "-" * 70
            print(string)
        return (expected_return, volatility, sharpe)
