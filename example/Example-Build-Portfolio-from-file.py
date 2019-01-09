# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # This example shows how to build a portfolio with the provided function `buildPortfolio()`
# ## In this example with data obtained from data files.
# 
# Note: The stock data is provided in two data files. The stock data was previously pulled from quandl.

# <markdowncell>

# ## Getting stock data

# <codecell>

import pandas as pd
import datetime

# <codecell>

# importing QPY's function to automatically build the portfolio
from qpy.portfolio import buildPortfolio

# <markdowncell>

# ## Get data from disk/file
# Here we use `pandas.read_cvs()` method to read in the data.

# <codecell>

# stock data was previously pulled from quandl and stored in ex1-stockdata.csv
# commands used to save data:
# # write data to disk:
# pf.getPortfolio().to_csv("ex1-portfolio.csv", encoding='utf-8', index=False, header=True)
# pf.getPfStockData().to_csv("ex1-stockdata.csv", encoding='utf-8', index=True, index_label="Date")
# read data from files:
df_pf = pd.read_csv("../data/ex1-portfolio.csv")
df_data = pd.read_csv("../data/ex1-stockdata.csv", index_col='Date', parse_dates=True)

# <markdowncell>

# ## Examining the DataFrames

# <codecell>

df_pf

# <codecell>

df_data.head(3)

# <markdowncell>

# ## Building a portfolio with `buildPortfolio()`
# `buildPortfolio()` is an interface that can be used in three different ways. One of which is shown below. For more information the docstring is shown below as well.
# 
# In this example buildPortfolio is being passed stock_data, which was read in from file above.

# <codecell>

print(buildPortfolio.__doc__)

# <codecell>

# building a portfolio by providing stock data
pf = buildPortfolio(df_pf, stock_data=df_data)

# <markdowncell>

# ## Examining the portfolio

# <codecell>

pf.portfolio

# <codecell>

pf.getPfStockData().head(2)

# <codecell>

df = pf.getPfStockData()

# <markdowncell>

# ### Extracting stock data by date
# Since quandl provides a DataFrame with an index of dates, it is easy to extract data from the portfolio for a given time frame. Three examples are shown below.

# <codecell>

df.loc[str(datetime.datetime(2015,1,2))]

# <codecell>

df.loc[df.index>datetime.datetime(2016,1,2)].head(3)

# <codecell>

df.loc[df.index.year==2017].head(3)

# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>

import numpy as np

#take log return
#goog = pf.getPfStockData()#['GOOG - Adj. Close'].to_frame()

goog = pf.getStock('GOOG').getStockData().copy(deep=True)#['GOOG - Adj. Close']
goog
#goog['test'] = goog.pct_change(1)
goog['return'] = goog['GOOG - Adj. Close'].pct_change(1).fillna(0)
# log return:
goog['log-return1'] = np.log(goog['GOOG - Adj. Close']) - np.log(goog['GOOG - Adj. Close'].shift(1)).fillna(0)
goog['log-return2'] = np.log(1 + goog['return']).fillna(0)
goog['log-return3'] = np.log(goog['GOOG - Adj. Close']).diff().fillna(0)

goog

for i in range(1,4):
    print(all(goog['log-return2'] - goog['log-return'+str(i)] < 1e-15))
goog.head(3)

# <codecell>



# <codecell>



# <codecell>



# <codecell>

pf.getStock('GOOG').getStockData().head(3)

'Adj. Close' in pf.getStock('GOOG').getStockData().columns
pf.getStock('GOOG').getStockData().columns

# <codecell>



# <codecell>


