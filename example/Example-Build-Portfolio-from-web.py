# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Building a portfolio with data from `quandl`/`yfinance`
# ## Building a portfolio with `build_portfolio()` by downloading relevant data through `quandl`/`yfinance` with stock names, start and end date and column labels
# This example only focuses on how to use `build_portfolio()` to get an instance of `Portfolio` by providing minimal information that is passed on to `quandl`/`yfinance`. For a more exhaustive description of this package and example, please try `Example-Analysis` and `Example-Optimisation`.

# <codecell>

import pandas as pd
import datetime

# importing some custom functions/objects
from finquant.portfolio import build_portfolio

# <markdowncell>

# ## Get data from `quandl`/`yfinance` and build portfolio
# First we need to build a pandas.DataFrame that holds relevant data for our portfolio. The minimal information needed are stock names and the amount of money to be invested in them, e.g. Allocation.

# <codecell>

# To play around yourself with different stocks, here is a short list of companies and their tickers
# d = {0: {'Name':'WIKI/GOOG', 'Allocation':20},  # Google
#      1: {'Name':'WIKI/AMZN', 'Allocation':33},  # Amazon
#      2: {'Name':'WIKI/MSFT', 'Allocation':18},  # Microsoft
#      3: {'Name':'WIKI/AAPL', 'Allocation':10},  # Apple
#      4: {'Name':'WIKI/KO', 'Allocation':15},    # Coca-Cola
#      5: {'Name':'WIKI/XOM', 'Allocation':11},   # Exxon Mobil
#      6: {'Name':'WIKI/JPM', 'Allocation':21},   # JP Morgan
#      7: {'Name':'WIKI/DIS', 'Allocation':9},    # Disney
#      8: {'Name':'WIKI/MCD', 'Allocation':23},   # McDonald's
#      9: {'Name':'WIKI/WMT', 'Allocation':3},    # Walmart
#     10: {'Name':'WIKI/YHOO', 'Allocation':7},   # Yahoo
#     11: {'Name':'WIKI/GS', 'Allocation':9},     # Goldman Sachs
#     }

# <codecell>

d = {
    0: {"Name": "WIKI/GOOG", "Allocation": 20},
    1: {"Name": "WIKI/AMZN", "Allocation": 10},
    2: {"Name": "WIKI/MCD", "Allocation": 15},
    3: {"Name": "WIKI/DIS", "Allocation": 18},
}
# If you wish to use Yahoo Finance as source, you must remove "WIKI/" from the stock names/tickers

pf_allocation = pd.DataFrame.from_dict(d, orient="index")

# <markdowncell>

# ### User friendly interface to quandl/yfinance
# As mentioned above, in this example `build_portfolio()` is used to build a portfolio by performing a query to `quandl`/`yfinance`.
#
# To download Google's stock data, `quandl` requires the string `"WIKI/GOOG"`. For simplicity, `FinQuant` facilitates a set of functions under the hood to sort out lots of specific commands/required input for `quandl`/`yfinance`. When using `FinQuant`, the user simply needs to provide a list of stock names/tickers.
# For example, if using `quandl` as a data source (default), a list of names/tickers as shown below is a valid input for `FinQuant`'s function `build_portfolio(names=names)`:
#  * `names = ["WIKI/GOOG", "WIKI/AMZN"]`
#
# If using `yfinance` as a data source, `FinQuant`'s function `build_portfolio(names=names)` expects the stock names to be without any leading/trailing string (check Yahoo Finance for correct stock names):
#  * `names = ["GOOG", "AMZN"]`
#
# By default, `FinQuant` uses `quandl` to obtain stock price data. The function `build_portfolio()` can be called with the optional argument `data_api` to use `yfinance` instead:
#  * `build_portfolio(names=names, data_api="yfinance")`
#
# In the below example we are using the default option, `quandl`.

# <codecell>

# here we set the list of names based on the names in
# the DataFrame pf_allocation
names = pf_allocation["Name"].values.tolist()

# dates can be set as datetime or string, as shown below:
start_date = datetime.datetime(2015, 1, 1)
end_date = "2017-12-31"

# While quandl/yfinance will download lots of different prices for each stock,
# e.g. high, low, close, etc, FinQuant will extract the column "Adj. Close" ("Adj Close" if using yfinance).

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
# As mentioned above, this example only shows how to use `build_portfolio()` to get an instance of `Portfolio` by downloading data through `quandl`/`yfinance`.
