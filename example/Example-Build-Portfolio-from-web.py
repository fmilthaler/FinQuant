# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Building a portfolio with data from `quandl`/`yfinance`
# ## Building a portfolio with `build_portfolio()` by downloading relevant data through `quandl`/`yfinance` with stock names, start and end date and column labels
# This example only focuses on how to use `build_portfolio()` to get an instance of `Portfolio` by providing a few items of information that is passed on to `quandl`/`yfinance`. For a more exhaustive description of this package and example, please try `Example-Analysis` and `Example-Optimisation`.

# <codecell>

import pandas as pd
import datetime

# importing some custom functions/objects
from finquant.portfolio import build_portfolio

# <markdowncell>

# ## Get data from `quandl`/`yfinance` and build portfolio
# First we need to build a pandas.DataFrame that holds relevant data for our portfolio. The minimal information needed are stock names and the amount of money to be invested in them, e.g. Allocation.

# <codecell>

# To play around yourself with different stocks, here is a short list of companies and their tickers on Yahoo Finance:
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
#     10: {'Name':'GS', 'Allocation':9},     # Goldman Sachs
#     }

# <codecell>

d = {
    0: {"Name": "GOOG", "Allocation": 20},
    1: {"Name": "AMZN", "Allocation": 10},
    2: {"Name": "MCD", "Allocation": 15},
    3: {"Name": "DIS", "Allocation": 18},
}
# If you wish to use `quandl` as source, you must add "WIKI/" at the beginning of stock names/tickers, as "WIKI/GOOG".

pf_allocation = pd.DataFrame.from_dict(d, orient="index")

# <markdowncell>

# ### User friendly interface to quandl/yfinance
# As mentioned above, in this example `build_portfolio()` is used to build a portfolio by performing a query to `quandl`/`yfinance`. We mention that `quandl` will be removed in future versions of `FinQuant` as it is deprecated.
#
# To download Google's stock data, `quandl` requires the string `"WIKI/GOOG"` and `yfinance` the string `"GOOG"`.
# For simplicity, `FinQuant` facilitates a set of functions under the hood to sort out lots of specific commands/required input for `quandl`/`yfinance`. When using `FinQuant`, the user simply needs to provide a list of stock names/tickers.
# For example, if using `quandl` as a data source (currently the default option), a list of names/tickers as shown below is a valid input for `FinQuant`'s function `build_portfolio(names=names)`:
#  * `names = ["WIKI/GOOG", "WIKI/AMZN"]`
#
# If using `yfinance` as a data source, `FinQuant`'s function `build_portfolio(names=names)` expects the stock names to be without any leading/trailing string (check Yahoo Finance for correct stock names):
#  * `names = ["GOOG", "AMZN"]`
#
# By default, `FinQuant` currently uses `quandl` to obtain stock price data. The function `build_portfolio()` can be called with the optional argument `data_api` to use `yfinance` instead:
#  * `build_portfolio(names=names, data_api="yfinance")`
#
# In the below example we are using `yfinance` to download stock data. We specify the start and end date of the stock prices to be downloaded.
# We also provide the optional parameter `market_index` to download the historical data of a market index. `FinQuant` can use them to calculate the beta parameter, measuring the portfolio's daily volatility compared to the market.

# <codecell>

# here we set the list of names based on the names in
# the DataFrame pf_allocation
names = pf_allocation["Name"].values.tolist()

# dates can be set as datetime or string, as shown below:
start_date = datetime.datetime(2015, 1, 1)
end_date = "2017-12-31"

# the market index used to compare the portfolio to (in this case S&P 500).
# If the parameter is omitted, no market comparison will be done
market_index = "^GSPC"

# While quandl/yfinance will download lots of different prices for each stock,
# e.g. high, low, close, etc, FinQuant will extract the column "Adj. Close" ("Adj Close" if using yfinance).

pf = build_portfolio(
    names=names,
    pf_allocation=pf_allocation,
    start_date=start_date,
    end_date=end_date,
    data_api="yfinance",
    market_index=market_index,
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
