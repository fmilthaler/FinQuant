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
pf = buildPortfolio(df_pf, data=df_data)

# <markdowncell>

# ## Portfolio is successfully built
# Getting data from the portfolio

# <codecell>

# the portfolio information DataFrame
print(pf.portfolio.name)
pf.portfolio

# <codecell>

# the portfolio stock data, prices DataFrame
pf.data.head(3)

# <markdowncell>

# ## Expected Return, Volatility and Sharpe Ratio of Portfolio
# The annualised expected return and volatility as well as the Sharpe Ratio are automatically computed. They are obtained as shown below.
# 
# The expected return and volatility are based on 252 trading days by default. The Sharpe Ratio is computed with a risk free rate of 0.005 by default.

# <codecell>

# expected (annualised) return
pf.expectedReturn

# <codecell>

# volatility
pf.volatility

# <codecell>

# Sharpe ratio (computed with a risk free rate of 0.005 by default)
pf.sharpe

# <markdowncell>

# ## Getting Skewness and Kurtosis of the stocks

# <codecell>

pf.skew

# <codecell>

pf.kurtosis

# <markdowncell>

# ## Nicely printing out portfolio quantities
# To print the expected annualised return, volatility, Sharpe ratio, skewness and Kurtosis of the portfolio and its stocks, one can simply do `pf.properties()`.

# <codecell>

print(pf)
pf.properties()

# <markdowncell>

# ## Daily returns and log returns
# `QPY` provides functions to compute daily returns and annualised mean returns of a given DataFrame in various ways.

# <codecell>

# annualised mean returns
pf.compPfMeanReturns()

# <codecell>

# daily returns (percentage change)
pf.compPfSimpleReturns().head(3)

# <codecell>

pf.compPfDailyLogReturns().head(3)

# <codecell>

# plotting stock data of portfolio
pf.data.plot()

# <markdowncell>

# The stock prices of Google and Amazon are much higher than those for McDonald's and Disney. Hence the fluctuations of the latter ones are barely seen in the above plot. One can use pandas.plot() method to create a secondary y axis.

# <codecell>

pf.data.plot(secondary_y = ["MCD", "DIS"], grid = True)

# <codecell>

# plotting returns (price_{t} / price_{t=0})
pf.compPfSimpleReturns().plot().axhline(y = 1, color = "black", lw = 3)

# <codecell>

# plotting daily percentage changes of returns
pf.compPfDailyReturns().plot().axhline(y = 0, color = "black")

# <codecell>

# plotting daily log returns
pf.compPfDailyLogReturns().plot().axhline(y = 0, color = "black")

# <markdowncell>

# ## Moving Averages
# `QPY` provides a method to compute moving averages. See below.

# <codecell>

# simple moving average
from qpy.moving_average import SMA, EWM
ax=pf.data.plot(secondary_y = ["KO", "MCD", "DIS"], grid = True)
SMA(pf.data, span=50).plot(ax=ax, secondary_y = ["KO", "MCD", "DIS"], grid = True)

# <codecell>

# exponential moving average
ax=pf.data.plot(secondary_y = ["MCD", "DIS"], grid = True)
EWM(pf.data).plot(ax=ax, secondary_y = ["KO", "MCD", "DIS"])

# <markdowncell>

# ### Many moving averages of one stock

# <codecell>

# get stock data for amazon
goog = pf.getStock("GOOG").data.copy(deep=True)

movingavg = [10, 50, 100, 200]
for i in movingavg:
    print("i = {}".format(i))
    goog[str(i)+"d"] = SMA(goog["GOOG"], span=i)

# <codecell>

goog.plot()

# <markdowncell>

# ## Portfolio optimisation
# Perform a Monte Carlo simulation to find the portfolio with the minimum volatility and maximum Sharpe Ratio.

# <codecell>

opt = pf.optimisePortfolio(num_trials=25000)
opt

# <markdowncell>

# ## Recomputing expected return, volatility and Sharpe ratio
# Note: When doing so, the instance variables are being reset by doing so.

# <codecell>

# If the return, volatility and Sharpe ratio need to be computed based on a different time window and/or risk free rate, one can recompute those values as shown below
freq = 100
rfr = 0.02
exret = pf.compPfExpectedReturn(freq=freq)
vol = pf.compPfVolatility(freq=freq)
sharpe = pf.compPfSharpe(riskFreeRate=rfr)
print("For {} trading days and a risk free rate of {}:".format(freq, rfr))
print("Expected return: {:0.3f}".format(exret))
print("Volatility: {:0.3f}".format(vol))
print("Sharpe Ratio: {:0.3f}".format(sharpe))

# <markdowncell>

# ## Extracting data of stocks individually
# Each stock (its information and data) of the portfolio is stored as a `Stock` data structure. If needed, one can of course extract the relevant data from the portfolio DataFrame, or access the `Stock` instance. The commands are very similar to the once for `Portfolio`. See below how it can be used.

# <codecell>

# getting Stock object from portfolio, for Google's stock
goog = pf.getStock("GOOG")
# getting the stock prices
goog_prices = goog.data
goog_prices.head(3)

# <codecell>

goog.compDailyReturns().head(3)

# <codecell>

goog.expectedReturn

# <codecell>

goog.volatility

# <codecell>

goog.skew

# <codecell>

goog.kurtosis

# <codecell>

print(goog)
goog.properties()

# <markdowncell>

# ## Extracting stock data by date
# Since quandl provides a DataFrame with an index of dates, it is easy to extract data from the portfolio for a given time frame. Three examples are shown below.

# <codecell>

df = pf.data
df.loc[str(datetime.datetime(2015,1,2))]

# <codecell>

df.loc[df.index>datetime.datetime(2016,1,2)].head(3)

# <codecell>

df.loc[df.index.year==2017].head(3)
