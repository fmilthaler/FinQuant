# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Example: Portfolio optimisation
# 
# This example shows how `QPY` can be used to optimise a portfolio.
# 
# Two different approaches are implemented in `QPY`:
#  1. Efficient Frontier
#  2. Monte Carlo run
# 
# With the *Efficient Frontier* approach, the portfolio can be optimised for
#  - minimum volatility,
#  - maximum Sharpe ratio
#  - minimum volatility for a given expected return
#  - maximum Sharpe ratio for a given target volatility
# by performing a numerical solve to minimise/maximise an objective function.
# 
# Alternatively a *Monte Carlo* run of `n` trials can be performed to find the optimal portfolios for
#  - minimum volatility,
#  - maximum Sharpe ratio
# 
# The approach branded as *Efficient Frontier* should be the preferred method for reasons of computational effort and accuracy. The latter approach is only included for the sake of completeness, and creation of beautiful plots.
# 
# ## Visualisation
# Not only does `QPY` allow for the optimisation of a portfolio with the above mentioned methods and objectives, `QPY` also allows for the computation and visualisation of an *Efficient Frontier* and *Monte Carlo* run.
# 
# Let `pf` be an instance of `Portfolio`. The *Efficient Frontier* can be computed and visualised with `pf.ef_plot_efrontier()`. The optimal portfolios for *minimum volatility* and *maximum Sharpe ratio* can be visualised with `pf.ef_plot_optimal_portfolios()`. And if required, the individual stocks of the portfolio can be visualised with `pf.plot_stocks(show=False)`. An overlay of these three commands is shown below.
# 
# Finally, the entire result of a *Monte Carlo* run can also be visualised automatically by `QPY`. An example is shown below.

# <markdowncell>

# ### Getting stock data

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd
import datetime
# importing QPY's function to automatically build the portfolio
from qpy.portfolio import buildPortfolio

# <codecell>

# plotting style:
plt.style.use('seaborn-darkgrid')
#set line width
plt.rcParams['lines.linewidth'] = 2
#set font size for titles
plt.rcParams['axes.titlesize'] = 14
#set font size for labels on axes
plt.rcParams['axes.labelsize'] = 12
#set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 10
#set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 10
#set figure size
plt.rcParams['figure.figsize'] = (10, 6)

# <markdowncell>

# ### Get data from disk/file
# Here we use `pandas.read_cvs()` method to read in the data.

# <codecell>

# stock data was previously pulled from quandl and stored in ex1-stockdata.csv
# read data from files:
df_data = pd.read_csv("../data/ex1-stockdata.csv", index_col='Date', parse_dates=True)
# building a portfolio by providing stock data
pf = buildPortfolio(data=df_data)
print(pf)
pf.properties()

# <markdowncell>

# # Portfolio optimisation
# ## Efficient Frontier
# Based on the **Efficient Frontier**, the portfolio can be optimised for
#  - minimum volatility
#  - maximum Sharpe ratio
#  - minimum volatility for a given target return
#  - maximum Sharpe ratio for a given target volatility
# 
# See below for an example for each optimisation.

# <codecell>

# if needed, change risk free rate and frequency/time window of the portfolio
print("pf.riskFreeRate = {}".format(pf.riskFreeRate))
print("pf.freq = {}".format(pf.freq))

# <codecell>

pf.ef_minimum_volatility()

# <codecell>

# optimisation for maximum Sharpe ratio
pf.ef_maximum_sharpe_ratio(verbose=True)

# <codecell>

# minimum volatility for a given target return of 0.26
pf.ef_efficient_return(0.26, verbose=True)

# <codecell>

# maximum Sharpe ratio for a given target volatility of 0.22
pf.ef_efficient_volatility(0.22)

# <markdowncell>

# ### Manually creating an instance of EfficientFrontier
# If required, or preferred, the below code shows how the same is achieved by manually creating an instance of EfficientFrontier, passing it the mean returns and covariance matrix of the previously assembled portfolio.

# <codecell>

from qpy.efficient_frontier import EfficientFrontier

# creating an instance of EfficientFrontier
ef = EfficientFrontier(pf.compMeanReturns(freq=1),
                       pf.compCov())
# optimisation for minimum volatility
ef.minimum_volatility()

# <codecell>

# printing out relevant quantities of the optimised portfolio
(expectedReturn, volatility, sharpe) = ef.properties(verbose=True)

# <markdowncell>

# ## Computing and visualising the Efficient Frontier
# `QPY` offers several ways to compute the *Efficient Frontier*.
#  1. Through the opject `pf`
#   - with automatically setting limits of the *Efficient Frontier*
#  2. By manually creating an instance of `EfficientFrontier` and providing the data from the portfolio
#   - with automatically setting limits of the *Efficient Frontier*
#   - by providing a range of target expected return values
# 
# As before, let `pf` and `ef` be instances of `Portfolio` and `EfficientFrontier` respectively. The following code snippets compute and plot an *Efficient Frontier* of a portfolio.
#  - `pf.ef_plot_efrontier()`
#  - ```targets = numpy.linspace(0.12, 0.45, 50)
#    efficient_frontier = ef.efficient_frontier(targets)
#    ef.plot_efrontier()
#    ```

# <markdowncell>

# ### Efficient frontier of `pf`

# <codecell>

# computing and plotting efficient frontier of pf
pf.ef_plot_efrontier(show=False)
# adding markers to optimal solutions
pf.ef.plot_optimal_portfolios()
# and adding the individual stocks to the plot
pf.plot_stocks(show=True)

# <markdowncell>

# ### Efficient Frontier with target return values

# <codecell>

import numpy as np
targets = np.linspace(0.12, 0.45, 50)
# computing efficient frontier
efficient_frontier = ef.efficient_frontier(targets)
# plotting efficient frontier
ef.plot_efrontier(show=False)
# adding markers to optimal solutions
pf.ef.plot_optimal_portfolios()
# and adding the individual stocks to the plot
pf.plot_stocks(show=True)

# <markdowncell>

# # Portfolio optimisation
# ## Monte Carlo
# Perform a Monte Carlo simulation to find the portfolio with the minimum volatility and maximum Sharpe Ratio.

# <codecell>

opt_w, opt_res = pf.mc_optimisation(num_trials=5000,
                                             verbose=True,
                                             plot=True)

# again, the individual stocks can be added to the plot
pf.plot_stocks(show=False)

# <codecell>

opt_res

# <codecell>

opt_w

# <markdowncell>

# # Optimisation overlay
# ## Overlay of Monte Carlo portfolios and Efficient Frontier solutions

# <codecell>

opt_w, opt_res = pf.mc_optimisation(num_trials=5000,
                                      verbose=False,
                                      plot=True)
pf.ef_plot_efrontier(show=False)
pf.ef.plot_optimal_portfolios()
pf.plot_stocks(show=False)
