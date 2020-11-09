# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Building a portfolio with data from `quandl`/`yfinance`
# ## Building a portfolio with `build_portfolio()` by downloading relevant data through `quandl`/`yfinance` with stock names, start and end date and column labels
# This example only focuses on how to use `build_portfolio()` and cluster stocks based on the return and volatility
# <codecell>

import datetime
import matplotlib.pyplot as plt

# importing some custom functions/objects
from finquant.portfolio import build_portfolio

# <markdowncell>

# ## Get data from `quandl`/`yfinance` and build portfolio
# First we need to list stocks we want to cluster.

# <codecell>

tickers_list = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'INTC', 'NVDA', 'ADBE',
                'PYPL', 'CSCO', 'NFLX', 'PEP', 'TSLA', 'CMCSA', 'AMGN', 'COST', 'TMUS', 'AVGO',
                'TXN', 'CHTR', 'QCOM', 'GILD', 'SBUX', 'INTU', 'VRTX', 'MDLZ', 'ISRG', 'FISV',
                'BKNG', 'ADP', 'REGN', 'ATVI', 'AMD', 'JD', 'MU', 'AMAT', 'ILMN', 'ADSK',
                'CSX', 'MELI', 'LRCX', 'ADI', 'ZM', 'BIIB', 'EA', 'KHC', 'WBA', 'LULU',
                'EBAY', 'MNST', 'DXCM', 'EXC', 'BIDU', 'XEL', 'WDAY', 'DOCU', 'SPLK', 'ORLY',
                'NXPI', 'CTSH', 'KLAC', 'SNPS', 'SGEN', 'ASML', 'IDXX', 'CSGP', 'CTAS', 'VRSK',
                'MAR', 'CDNS', 'PAYX', 'ALXN', 'MCHP', 'SIRI', 'ANSS', 'VRSN', 'FAST', 'BMRN',
                'XLNX', 'INCY', 'DLTR', 'SWKS', 'ALGN', 'CERN', 'CPRT', 'CTXS', 'TTWO', 'MXIM',
                'CDW', 'CHKP', 'WDC', 'ULTA', 'NTAP', 'FOXA', 'LBTYK']

# <codecell>

#tickers_list = ['AAPL', 'MSFT', 'AMZN']

# dates can be set as datetime:
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2020, 1, 1)

# While quandl/yfinance will download lots of different prices for each stock,
# e.g. high, low, close, etc, FinQuant will extract the column "Adj. Close" ("Adj Close" if using yfinance).

pf = build_portfolio(
    names=tickers_list, start_date=start_date, end_date=end_date, data_api='yfinance'
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
pf.cluster_stocks(n_clusters=10)

plt.show()

# ## Please continue with `Example-Build-Portfolio-from-file.py`.
# As mentioned above, this example only shows how to use `build_portfolio()` to get an instance of `Portfolio` by downloading data through `quandl`/`yfinance`.
