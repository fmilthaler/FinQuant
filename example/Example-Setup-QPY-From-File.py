# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# ## Getting stock data

# <codecell>

import numpy as np
import pandas as pd
import datetime
import os

# importing some custom functions/objects
from qpy.portfolio import Portfolio, Stock, buildPortfolio

# <markdowncell>

# ## Get data disk/file

# <codecell>

# stock data was previously pulled from quandl and stored in ex1-stockdata.csv
pf_info = pd.read_csv("../data/ex1-portfolio.csv")
pfstockdata = pd.read_csv("../data/ex1-stockdata.csv", index_col='Date', parse_dates=True)

# <codecell>

pf_info

# <codecell>

pfstockdata.head(3)

# <codecell>

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2017,12,31)
datacolumns = ['Adj. Close']
datacolumns = ['Adj. Close', 'High']
#pf = buildPortfolioFromDf(pf_info, stock_data=pfstockdata)
#pf = buildPortfolio(pf_info, names=['GOOG','MSFT'], 
#                    start_date=start, end_date=end, 
#                    #datacolumns=datacolumns,
#                    roi_data='test',
#                   )

pf = buildPortfolio(pf_info, #names=['GOOG','MSFT'],
                    stock_data=pfstockdata,
                    #start_date=1
                    #datacolumns=datacolumns
                   )

#pf = buildPortfolio(pf_info, names=['GOOG','MSFT'],
#                    roi_data=pfstockdata,
#                   )

# <codecell>

print(pf)
pf.getPfStockData().head(2)
#pf.getPfStockData().tail(2)

# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>


