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

# ## Get data from disk/file

# <codecell>

# stock data was previously pulled from quandl and stored in ex1-stockdata.csv
# commands used to save data: 
# # write data to disk:
# pf.getPortfolio().to_csv("ex1-portfolio.csv", encoding='utf-8', index=False, header=True)
# pf.getPfStockData().to_csv("ex1-stockdata.csv", encoding='utf-8', index=True, index_label="Date")
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

#print(buildPortfolio.__doc__)

# <codecell>



# <codecell>

#assets = ['AAPL',       # Apple
#          'KO',         # Coca-Cola
#          'DIS',        # Disney
#          'XOM',        # Exxon Mobil
#          'JPM',        # JPMorgan Chase
#          'MCD',        # McDonald's
#          'WMT']        # Walmart

# <codecell>

df = pf.getPfStockData()

# <codecell>

#df.query("Date=="+str(datetime.datetime(2015,1,2)))
df.loc[str(datetime.datetime(2015,1,2))]

# <codecell>

#df.loc[df.index>datetime.datetime(2015,1,2)]
df.loc[df.index.year==2017]

# <codecell>

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


