# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Example:
# ## Building a portfolio with `build_portfolio()` with data obtained from data files.
# Note: The stock data is provided in two data files. The stock data was previously pulled from quandl.

# <codecell>

import datetime
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

# importing FinQuant's function to automatically build the portfolio
from finquant.portfolio import build_portfolio

# <codecell>

# plotting style:
plt.style.use("seaborn-v0_8-darkgrid")
# set line width
plt.rcParams["lines.linewidth"] = 2
# set font size for titles
plt.rcParams["axes.titlesize"] = 14
# set font size for labels on axes
plt.rcParams["axes.labelsize"] = 12
# set size of numbers on x-axis
plt.rcParams["xtick.labelsize"] = 10
# set size of numbers on y-axis
plt.rcParams["ytick.labelsize"] = 10
# set figure size
plt.rcParams["figure.figsize"] = (10, 6)

# <markdowncell>

# ## Building a portfolio with `build_portfolio()`
# As in previous example, using `build_portfolio()` to generate an object of `Portfolio`.

# <codecell>

# read data from files:
df_data_path = pathlib.Path.cwd() / ".." / "data" / "ex1-stockdata.csv"
df_data = pd.read_csv(df_data_path, index_col="Date", parse_dates=True)
# building a portfolio by providing stock data
pf = build_portfolio(data=df_data)

# <markdowncell>

# ## Expected Return, Volatility, Sharpe Ratio, Sortino Ratio, and Value at Risk of Portfolio
# The annualised expected return and volatility, as well as the Sharpe Ratio, the Sortino Ratio, and Value at Risk are automatically computed.
# They are obtained as shown below.
# The expected return and volatility are based on 252 trading days by default.
# The Sharpe Ratio and the Sortino ratio are computed with a risk free rate of 0.005 by default.
# The Value at Risk is computed with a confidence level of 0.95 by default.

# <codecell>

# expected (annualised) return
print(pf.expected_return)

# <codecell>

# volatility
print(pf.volatility)

# <codecell>

# Sharpe Ratio (computed with a risk free rate of 0.005 by default)
print(pf.sharpe)

# <codecell>

# Sortino Ratio (computed with a risk free rate of 0.005 by default)
print(pf.sortino)

# <codecell>

# Value at Risk (computed with a confidence level of 0.95 by default)
print(pf.var)

# <markdowncell>

# ## Getting Skewness and Kurtosis of the stocks

# <codecell>

print(pf.skew)

# <codecell>

print(pf.kurtosis)

# <markdowncell>

# ## Nicely printing out portfolio quantities
# To print the expected annualised return, volatility, Sharpe Ratio, Sortino Ratio, skewness and Kurtosis of the portfolio and its stocks, one can simply do `pf.properties()`.

# <codecell>

print(pf)
pf.properties()

# <markdowncell>

# ## Daily returns and log returns
# `FinQuant` provides functions to compute daily returns and annualised mean returns of a given DataFrame in various ways.

# <codecell>

# annualised mean returns
print(pf.comp_mean_returns())

# <codecell>

# daily returns (percentage change)
print(pf.comp_cumulative_returns().head(3))

# <codecell>

print(pf.comp_daily_log_returns().head(3))

# <codecell>

# plotting stock data of portfolio
pf.data.plot()
plt.show()

# <markdowncell>

# The stock prices of Google and Amazon are much higher than those for McDonald's and Disney. Hence the fluctuations of the latter ones are barely seen in the above plot. One can use `pandas.plot()` method to create a secondary y axis.

# <codecell>

pf.data.plot(secondary_y=["WIKI/MCD", "WIKI/DIS"], grid=True)
plt.show()

# <codecell>

# plotting cumulative returns (price_{t} - price_{t=0}) / price_{t=0}
pf.comp_cumulative_returns().plot().axhline(y=0, color="black", lw=3)
plt.show()

# <codecell>

# plotting daily percentage changes of returns
pf.comp_daily_returns().plot().axhline(y=0, color="black")
plt.show()

# <codecell>

# plotting daily log returns
pf.comp_daily_log_returns().plot().axhline(y=0, color="black")
plt.show()

# <codecell>

# cumulative log returns
pf.comp_daily_log_returns().cumsum().plot().axhline(y=0, color="black")
plt.show()

# <markdowncell>

# ## Moving Averages
# `FinQuant` provides a module `finquant.moving_average` to compute moving averages. See below.

# <codecell>

from finquant.moving_average import sma

# simple moving average
ax = pf.data.plot(secondary_y=["WIKI/MCD", "WIKI/DIS"], grid=True)
# computing simple moving average over a span of 50 (trading) days
# and plotting it
sma(pf.data, span=50).plot(ax=ax, secondary_y=["WIKI/MCD", "WIKI/DIS"], grid=True)
plt.show()

# <codecell>

from finquant.moving_average import ema

# exponential moving average
ax = pf.data.plot(secondary_y=["WIKI/MCD", "WIKI/DIS"], grid=True)
# computing exponential moving average and plotting it
ema(pf.data).plot(ax=ax, secondary_y=["WIKI/MCD", "WIKI/DIS"])
plt.show()

# <markdowncell>

# ## Band of moving averages and Buy/Sell signals
# `FinQuant` also provides a method `finquant.moving_average.compute_ma` that automatically computes and plots several moving averages. It also **finds buy/sell signals based on crossovers** of the shortest and longest moving average.
# To learn more about it and its input arguments, read its docstring and see the example below.

# <codecell>

from finquant.moving_average import compute_ma

print(compute_ma.__doc__)

# <codecell>

# get stock data for disney
dis = pf.get_stock("WIKI/DIS").data.copy(deep=True)
# we want moving averages of 10, 50, 100, and 200 days.
spans = [10, 50, 100, 150, 200]
# compute and plot moving averages
dis_ma = compute_ma(dis, ema, spans, plot=True)
plt.show()

# <markdowncell>

# ## Plot the Bollinger Band of one stock
# The Bollinger Band can be automatically computed and plotted with the method `finquant.moving_average.plot_bollinger_band`. See below for an example.

# <codecell>

# plot the bollinger band of the disney stock prices
from finquant.moving_average import plot_bollinger_band

# get stock data for disney
dis = pf.get_stock("WIKI/DIS").data.copy(deep=True)
span = 20
# for simple moving average:
plot_bollinger_band(dis, sma, span)
plt.show()
# for exponential moving average:
plot_bollinger_band(dis, ema, span)
plt.show()

# <markdowncell>

# ## Recomputing expected return, volatility and Sharpe ratio
# **Note**: When doing so, the instance variables for
#  - Expected return
#  - Volatility
#  - Sharpe Ratio
# are automatically recomputed.

# <codecell>

# If the return, volatility and Sharpe ratio need to be computed based
# on a different time window and/or risk free rate, one can recompute
# those values as shown below
# 1. set the new value(s)
pf.freq = 100
pf.risk_free_rate = 0.02

# 2.a compute and get new values based on new freq/risk_free_rate
exret = pf.comp_expected_return(freq=100)
vol = pf.comp_volatility(freq=100)
sharpe = pf.comp_sharpe()
print(
    "For {} trading days and a risk free rate of {}:".format(pf.freq, pf.risk_free_rate)
)
print("Expected return: {:0.3f}".format(exret))
print("Volatility: {:0.3f}".format(vol))
print("Sharpe Ratio: {:0.3f}".format(sharpe))

# 2.b print out properties of portfolio (which is based on new freq/risk_free_rate)
pf.properties()

# <markdowncell>

# ## Extracting data of stocks individually
# Each stock (its information and data) of the portfolio is stored as a `Stock` data structure. If needed, one can of course extract the relevant data from the portfolio DataFrame, or access the `Stock` instance. The commands are very similar to the once for `Portfolio`. See below how it can be used.

# <codecell>

# getting Stock object from portfolio, for Google's stock
goog = pf.get_stock("WIKI/GOOG")
# getting the stock prices
goog_prices = goog.data
print(goog_prices.head(3))

# <codecell>

print(goog.comp_daily_returns().head(3))

# <codecell>

print(goog.expected_return)

# <codecell>

print(goog.volatility)

# <codecell>

print(goog.skew)

# <codecell>

print(goog.kurtosis)

# <codecell>

print(goog)
goog.properties()

# <markdowncell>

# ## Extracting stock data by date
# Since quandl provides a DataFrame with an index of dates, it is easy to extract data from the portfolio for a given time frame. Three examples are shown below.

# <codecell>

print(pf.data.loc[str(datetime.datetime(2015, 1, 2))])

# <codecell>

print(pf.data.loc[pf.data.index > datetime.datetime(2016, 1, 2)].head(3))

# <codecell>

print(pf.data.loc[pf.data.index.year == 2017].head(3))

# <markdowncell>

# ## Momentum Indicators
# `FinQuant` provides a module `finquant.momentum_indicators` to compute and
# visualize a number of momentum indicators. Currently RSI (Relative Strength Index)
# and MACD (Moving Average Convergence Divergence) indicators are available.
# See below.

# <codecell>
# plot the RSI (Relative Strength Index) for disney stock proces
from finquant.momentum_indicators import plot_relative_strength_index as rsi

# get stock data for disney
dis = pf.get_stock("WIKI/DIS").data.copy(deep=True)

# plot RSI - by default this plots RSI against the price in two graphs
rsi(dis)
plt.show()

# plot RSI with custom arguments
rsi(dis, oversold=20, overbought=80)
plt.show()

# plot RSI standalone graph
rsi(dis, oversold=20, overbought=80, standalone=True)
plt.show()

# <codecell>
# plot MACD for disney stock proces
from finquant.momentum_indicators import plot_macd

# using short time frame of data due to plot warnings from matplotlib/mplfinance
dis = dis[0:300]

# plot MACD - by default this plots RSI against the price in two graphs
plot_macd(dis)
plt.show()

# plot MACD using custom arguments
plot_macd(dis, longer_ema_window=30, shorter_ema_window=15, signal_ema_window=10)
plt.show()
