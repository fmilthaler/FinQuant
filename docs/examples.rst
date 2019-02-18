.. _examples:

########
Examples
########


For more information about the project and details on how to use it, please
look at the examples provided in ``./example``:
 - ``./example/Example-Build-Portfolio-with-quandl.py``
 - ``./example/Example-Build-Portfolio-from-file.py``
 - ``./example/Example-Analysis.py``
 - ``./example/Example-Optimisation.py``

.. note:: In the below examples, ``pf`` refers to an instance of ``finquant.portfolio.Portfolio``, the object that holds all stock prices and computes its most common quantities automatically. To make *FinQuant* a user-friendly program, that combines data analysis, visualisation and optimisation, the object also provides interfaces to the main features that are provided in the modules in ``./finquant/`` and are discussed throughout this documentation.


Building a portfolio with *quandl*
==================================
``./example/Example-Build-Portfolio-with-quandl.py``: Shows how to use *FinQuant* to build a financial portfolio by downloading stock price data through the Python package *quandl*.


Building a portfolio with preset data
=====================================
``./example/Example-Build-Portfolio-from-file.py``: Shows how to use *FinQuant* to build a financial portfolio by providing stock price data yourself, e.g. by reading data from disk/file.


Analysis of a portfolio
=======================
``./example/Example-Analysis.py``: This example shows how to use an instance of ``finquant.portfolio.Portfolio``, get the portfolio's quantities, such as

- Expected Returns,
- Volatility,
- Sharpe Ratio.

It also shows how to extract individual stocks from the given portfolio. Moreover it shows how to compute and visualise:

- the different Returns provided by the module ``finquant.returns``,
- *Moving Averages*, a band of *Moving Averages*, and a *Bollinger Band*.


Optimisation of a portfolio
===========================
``./example/Example-Optimisation.py``: This example focusses on the optimisation of a portfolio. To achieve this, the example shows the usage of ``finquant.efficient_frontier.EfficientFrontier`` for numerically optimising the portfolio, for the

- Minimum Volatility
- Maximum Sharpe Ratio
- Minimum Volatility for a given target Return
- Maximum Sharpe Ratio for a given target Volatility.

Furthermore, it is also shown how the entire *Efficient Frontier* and the optimal portfolios can be computed and visualised. If needed, it also gives an example of plotting the individual stocks of the given portfolio within the computed *Efficient Frontier*.

Also, the optimisation of a portfolio and its visualisation based on a *Monte Carlo* is shown.

Finally, *FinQuant*'s visualisation methods allow for overlays, if this is desired. Thus, with only the following few lines of code, one can create an overlay of the *Monte Carlo* run, the *Efficient Frontier*, its optimised portfolios for *Minimum Volatility* and *Maximum Sharpe Ratio*, as well as the portfolio's individual stocks.

