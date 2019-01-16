# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Example: Portfolio optimisation
# 
# Note: The stock data is provided in two data files. The stock data was previously pulled from quandl.

# <markdowncell>

# ### Getting stock data

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd
import datetime

# <codecell>

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
df_pf = pd.read_csv("../data/ex1-portfolio.csv")
df_data = pd.read_csv("../data/ex1-stockdata.csv", index_col='Date', parse_dates=True)

# <codecell>

# building a portfolio by providing stock data
pf = buildPortfolio(df_pf, data=df_data)

# <codecell>

print(pf)
pf.properties()

# <markdowncell>

# # Portfolio optimisation
# ## Monte Carlo
# Perform a Monte Carlo simulation to find the portfolio with the minimum volatility and maximum Sharpe Ratio.

# <codecell>

opt_w, opt_res = pf.optimisePortfolio(num_trials=5000,
                                      verbose=True,
                                      plot=True)

# <codecell>

opt_res

# <codecell>

opt_w

# <markdowncell>

# # Portfolio optimisation
# ## Efficient Frontier
# Based on the __Efficient Frontier__, the portfolio can be optimised for
#  - minimum volatility
#  - maximum Sharpe ratio
#  - minimum volatility for a given target return
#  - maximum Sharpe ratio for a given target volatility
# 
# See below for an example for each optimisation.

# <codecell>

from qpy.efficient_frontier import EfficientFrontier

# creating an instance of EfficientFrontier
ef = EfficientFrontier(pf.compMeanReturns(freq=1),
                       pf.compCov())
# optimisation for maximum Sharpe ratio
ef.maximum_sharpe_ratio()

# <codecell>

# printing out relevant quantities of the optimised portfolio
(expectedReturn, volatility, sharpe) = ef.properties()

# <codecell>

# minimum volatility
ef.minimum_volatility()
ef.properties()

# <codecell>

# minimum volatility for a given target return of 0.26
ef.efficient_return(0.26)
ef.properties()

# <codecell>

# maximum Sharpe ratio for a given target volatility of 0.22
ef.efficient_return(0.22)
ef.properties()

# <markdowncell>

# # Computing and visualising the Efficient Frontier

# <codecell>

import numpy as np
targets = np.linspace(0.12, 0.45, 50)

# computing efficient frontier
efficient_frontier = ef.efficient_frontier(targets)
# plotting efficient frontier
ef.plot_efrontier(show=False)

# adding stocks of the portfolio to the plot
stock_returns = pf.compMeanReturns()
stock_volatility = pf.compStockVolatility()

# adding stocks of the portfolio to the plot
# plot stocks individually:
plt.scatter(stock_volatility,
            stock_returns,
            marker='o',
            s=200)
# adding text to stocks in plot:
for i, txt in enumerate(stock_returns.index):
    plt.annotate(txt,
                 (stock_volatility[i], stock_returns[i]),
                 xytext=(10,0),
                 textcoords='offset points',
                 label=i)

plt.legend(['efficient frontier', 'Stocks'])


# <codecell>



# <codecell>



# <codecell>



# <codecell>


