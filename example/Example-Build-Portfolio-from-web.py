# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Building a portfolio with data from `quandl`
# ## Building a portfolio with `build_portfolio()` by downloading relevant data through quandl with stock names, start and end date and column labels
# This example only focuses on how to use `build_portfolio()` to get an instance of `Portfolio` by providing minimal information that is passed on to `quandl`. For a more exhaustive description of this package and example, please try `Example-Analysis` and `Example-Optimisation`.

# <codecell>

import pandas as pd
import datetime
# importing some custom functions/objects
from finquant.portfolio import build_portfolio

# <markdowncell>

# ## Get data from quandl and build portfolio
# First we need to build a pandas.DataFrame that holds relevant data for our portfolio. The minimal information needed are stock names and the amount of money to be invested in them, e.g. Allocation.

# <codecell>

# To play around yourself with different stocks, here is a list of companies and their tickers
# d = {0: {'Name':'GOOG', 'Allocation':20},  # Google
#      1: {'Name':'AMZN', 'Allocation':33},  # Amazon
#      2: {'Name':'MSFT', 'Allocation':18},  # Microsoft
#      3: {'Name':'AAPL', 'Allocation':10},  # Apple
#      4: {'Name':'KO', 'Allocation':15},    # Coca-Cola
#      5: {'Name':'XOM', 'Allocation':11},   # Exxon Mobil
#      6: {'Name':'JPM', 'Allocation':21},   # JP Morgan
#      7: {'Name':'DIS', 'Allocation':9},    # Disney
#      8: {'Name':'MCD', 'Allocation':23},   # McDonald's
#      9: {'Name':'WMT', 'Allocation':3},    # Walmart
#     10: {'Name':'YHOO', 'Allocation':7},   # Yahoo
#     11: {'Name':'GS', 'Allocation':9},     # Goldman Sachs
#     }

# <codecell>

d = {
    0: {"Name": "GOOG", "Allocation": 20},
    1: {"Name": "AMZN", "Allocation": 10},
    2: {"Name": "MCD", "Allocation": 15},
    3: {"Name": "DIS", "Allocation": 18},
}
pf_allocation = pd.DataFrame.from_dict(d, orient="index")

# <markdowncell>

# ### User friendly interface to quandl
# As mentioned above, in this example `build_portfolio()` is used to build a portfolio by performing a query to `quandl`.
# To download Google's stock data, `quandl` requires the string `"WIKI/GOOG"`. For simplicity, `FinQuant` facilitates a set of functions under the hood to sort out lots of specific commands/required input for `quandl`. When using `FinQuant`, the user simply needs to provide a list of stock names/tickers. Moreover, the leading `"WIKI/"` in `quandl`'s request can be set by the user or not.
# For example, all three lists of tickers/names as shown below are valid input for
# `FinQuant`'s function `build_portfolio(names=names)`:
#  * `names = ['WIKI/GOOG', 'WIKI/AMZN']`
#  * `names = ['GOOG', 'AMZN']`
#  * `names = ['WIKI/GOOG', 'AMZN']`

# <codecell>

# here we set the list of names based on the names in
# the DataFrame pf_allocation
names = pf_allocation["Name"].values.tolist()

# dates can be set as datetime or string, as shown below:
start_date = datetime.datetime(2015, 1, 1)
end_date = "2017-12-31"

# While quandl will download lots of different prices for each stock,
# e.g. high, low, close, etc, FinQuant will extract the column "Adj. Close".

pf = build_portfolio(
    names=names, pf_allocation=pf_allocation, start_date=start_date, end_date=end_date
)

# <markdowncell>

# ## Portfolio is successfully built
# Getting data from the portfolio

# <codecell>

# the portfolio information DataFrame
print(pf.portfolio)

# <codecell>

# the portfolio stock data, prices DataFrame
print(pf.data.head(3))

# <codecell>

# print out information and quantities of given portfolio
print(pf)
pf.properties()

# <markdowncell>

# ## Please continue with `Example-Build-Portfolio-from-file.py`.
# As mentioned above, this example only shows how to use `build_portfolio()` to get an instance of `Portfolio` by downloading data through `quandl`.
