# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Example:
# ## Building a portfolio with `buildPortfolio()` by downloading relevant data through quandl with stock names, start and end date and column labels

# <codecell>

import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd

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

# <codecell>

import datetime

# <codecell>

# importing some custom functions/objects
#from qpy.portfolio import Portfolio, Stock, buildPortfolio
from qpy.portfolio import buildPortfolio

# <codecell>

import quandl
import os

# <codecell>

filename = os.environ['HOME']+'/.quandl/api_key'
with open(filename) as f:
    lines = f.readlines()
if (len(lines) != 1):
    raise ValueError('Could not get the quandl api key from '+filename)
quandl_api_key = ''.join(lines).strip()

# setting api key:
quandl.ApiConfig.api_key = quandl_api_key

# <markdowncell>

# ## Get data from quandl and build portfolio

# <codecell>

# To play around yourself with different stocks, here is a list of companies and their tickers
# d = {0 : {'Name':'GOOG', 'FMV':20}, # Google
#      1: {'Name':'AMZN', 'FMV':33},  # Amazon
#      2: {'Name':'MSFT', 'FMV':18},  # Microsoft
#      3: {'Name':'AAPL', 'FMV':10},  # Apple
#      4: {'Name':'KO', 'FMV':15},    # Coca-Cola
#      5: {'Name':'XOM', 'FMV':11},   # Exxon Mobil
#      6: {'Name':'JPM', 'FMV':21},   # JP Morgan
#      7: {'Name':'DIS', 'FMV':9},    # Disney
#      8: {'Name':'MCD', 'FMV':23},   # McDonald's
#      9: {'Name':'WMT', 'FMV':3},    # Walmart
#     10: {'Name':'YHOO', 'FMV':7},   # Yahoo
#     11: {'Name':'GS', 'FMV':9},     # Goldman Sachs
#     }
# pf_information = pd.DataFrame.from_dict(d, orient='index')

# <codecell>

d = {0 : {'Name':'GOOG', 'FMV':20},
     1: {'Name':'AMZN', 'FMV':33},
     2: {'Name':'MCD', 'FMV':15},
     3: {'Name':'DIS', 'FMV':9},
    }
pf_information = pd.DataFrame.from_dict(d, orient='index')

# <markdowncell>

# ### User friendly interface to quandl
# As mentioned above, in this example `buildPortfolio()` is used to build a portfolio by performing a query through `quandl`.
# 
# To download Google's stock data, `quandl` requires the string "WIKI/GOOG". For simplicity, `QPY` facilitates a set of functions under the hood to sort out lots of specific commands/required input for `quandl`. When using `QPY`, the user simply needs to provide a list of stock names/tickers. Moreover, the leading `"WIKI/"` in `quandl`'s request can be set by the user or not.
# 
# For example, all three lists as shown below are valid input for
# `QPY`'s function `buildPortfolio(pfinfo, names=names)`:
#  * `names = ['WIKI/GOOG', 'WIKI/AMZN']`
#  * `names = ['GOOG', 'AMZN']`
#  * `names = ['WIKI/GOOG', 'AMZN']`

# <codecell>

# here we set the list of names based on the names in
# the DataFrame pf_information
names = pf_information['Name'].values.tolist()

# dates can be set as datetime or string, as shown below:
start_date = datetime.datetime(2015,1,1)
end_date = '2017-12-31'

# The user can also provide a list of column labels to extract from
# the data obtained from quandl. If none is set, QPY will extract
# the column 'Adj. Close' for each stock.
datacolumns = ['Adj. Close', 'High', 'Low']

pf = buildPortfolio(pf_information,
                    names=names,
                    start_date=start_date,
                    end_date=end_date,
                    datacolumns=datacolumns)

# <codecell>

pf_information

# <codecell>

pf.getPfStockData().head(3)

# <codecell>


# <markdowncell>

# ## Portfolio optimisation

# <codecell>

# pf.optimisePortfolio(1000000, plot=True)
opt = pf.optimisePortfolio()
opt
