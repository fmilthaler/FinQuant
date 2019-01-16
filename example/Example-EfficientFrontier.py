# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Example:
# ## Building a portfolio with `buildPortfolio()` with data obtained from data files.
# 
# Note: The stock data is provided in two data files. The stock data was previously pulled from quandl.

# <markdowncell>

# ## Getting stock data

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

# ## Get data from disk/file
# Here we use `pandas.read_cvs()` method to read in the data.

# <codecell>

# stock data was previously pulled from quandl and stored in ex1-stockdata.csv
# commands used to save data:
# pf.getPortfolio().to_csv("ex1-portfolio.csv", encoding='utf-8', index=False, header=True)
# pf.getPfStockData().to_csv("ex1-stockdata.csv", encoding='utf-8', index=True, index_label="Date")
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

# ## Portfolio optimisation
# Perform a Monte Carlo simulation to find the portfolio with the minimum volatility and maximum Sharpe Ratio.

# <codecell>

opt_w, opt_res = pf.optimisePortfolio(num_trials=5000,
                                      verbose=True,
                                      plot=True)

# <codecell>

opt_res

# <codecell>

opt_w

# <codecell>

from qpy.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(pf.compMeanReturns(freq=1),
                       pf.compCov())
ef.maximum_sharpe_ratio()

# <codecell>

ef.properties()

# <codecell>

# minimum volatility
ef.minimum_volatility()
ef.properties()

# <codecell>

ef.efficient_return(1.0)
ef.properties()

# <codecell>

ef.efficient_return(0.2)

# <codecell>

ef.properties()

# <codecell>



# <codecell>

ef.efficient_volatility(1, riskFreeRate=0.005)

# <codecell>

ef.properties()

# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>


