# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# ## Getting stock data

# <codecell>

import matplotlib.pyplot as plt
# to plot within the notebook:
%matplotlib inline
#pylab

import math, random
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
#from scipy import stats

sns.set_style('darkgrid')

# <codecell>

##set line width
plt.rcParams['lines.linewidth'] = 2
##set font size for titles 
plt.rcParams['axes.titlesize'] = 14
##set font size for labels on axes
plt.rcParams['axes.labelsize'] = 12
##set size of numbers on x-axis
plt.rcParams['xtick.labelsize'] = 10
##set size of numbers on y-axis
plt.rcParams['ytick.labelsize'] = 10

# <codecell>

import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

# <codecell>

# importing some custom functions/objects
from qpy.portfolio import Portfolio, Stock, buildPortfolioFromQuandl

# <codecell>

import quandl
import os

# <codecell>

#filename = os.environ['HOME']+'/.quandl/api_key'
#with open(filename) as f:
#    lines = f.readlines()
#if (len(lines) != 1):
#    raise ValueError('Could not get the quandl api key from '+filename)
#quandl_api_key = ''.join(lines).strip()

## setting api key:
#quandl.ApiConfig.api_key = quandl_api_key

# <markdowncell>

# ## Get data from quandl

# <codecell>

d = {0 : {'Name':'GOOG', 'FMV':20}, 1: {'Name':'AMZN', 'FMV':33}, 2: {'Name':'MSFT', 'FMV':8}}
pfinfo = pd.DataFrame.from_dict(d, orient='index')

# <codecell>

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2017,12,31)
#names = ['WIKI/AAPL', 'WIKI/MSFT']
names = [d[k]['Name'] for k, v in d.items()]
datacolumns = ['Adj. Close']
datacolumns = ['Adj. Close', 'High']

pf = buildPortfolioFromQuandl(pfinfo, names, start, end, datacolumns)
pfinfo

# <codecell>

print(pf)

# <codecell>

pf.getPfStockData().head(3)
pf.getStocks()['AMZN'].getStockData().head(3)

# <codecell>

pf.getPfStockData().tail(2)

# <codecell>

goog = pf.getStock('GOOG')
print(goog)
goog.getStockData().head(2)
#pf.getPfStockData()

# <codecell>

for stockname in pf.getPfStockData().columns:
        print("stockname = ", stockname)

# <codecell>



# <codecell>



# <codecell>



# <markdowncell>

# ### Continue here

# <codecell>

# We will look at stock prices over a one year period, starting at January 1, 2017
#start = datetime.datetime(2015,1,1)
#end = datetime.datetime(2017,12,31)

names = ['WIKI/'+str(i) for i in ['AAPL', 'MSFT', 'GOOG', 'AMZN']]#+'WIKI'
print("names = ",names)

apple_orig = quandl.get('WIKI/AAPL', start_date=start, end_date=end)#[start:end]
microsoft_orig = quandl.get('WIKI/MSFT', start_date=start, end_date=end)#[start:end]
google_orig = quandl.get('WIKI/GOOG', start_date=start, end_date=end)#[start:end]
amazon_orig = quandl.get('WIKI/AMZN', start_date=start, end_date=end)#[start:end]

apple_orig.head()

# <codecell>

#data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'close'] }, ticker = ['AAPL', 'MSFT'],
#                        date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })
data = quandl.get(names, start_date=start, end_date=end)#['Adj. Close'].to_frame()
#data = quandl.get('WIKI/AAPL', start_date=start, end_date=end, collapse="monthly")['Adj. Close'].to_frame()
data.head()

# <codecell>

s=['WIKI/AAPL', 'WIKI/MSFT', 'WIKI/GOOG', 'WIKI/AMZN']
test = quandl.get(s, column='Adj. Close', start_date=start, end_date=end)
test.head()

# <codecell>

apple_orig.tail()

# <codecell>

# cutting dates:
apple = apple_orig.loc[start:end]
microsoft = microsoft_orig.loc[start:end]
google = google_orig.loc[start:end]
amazon = amazon_orig.loc[start:end]

# <codecell>

# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Adj. Close"],
                      "MSFT": microsoft["Adj. Close"],
                      "GOOG": google["Adj. Close"],
                      "AMZN": amazon["Adj. Close"]})

stocks.plot(grid = True)
stocks.head()

# <codecell>

# with second y axis on the right:
stocks.plot(secondary_y = ["AAPL", "MSFT"], grid = True)

# <markdowncell>

# A *better* solution, though, would be to plot the information we actually want: the stock's returns. This involves transforming the data into something more useful for our purposes. There are multiple transformations we could apply.
# 
# One transformation would be to consider the stock’s return since the beginning of the period of interest. In other words, we plot:
# 
# $\displaystyle\text{return}_{t,0} = \dfrac{\text{price}_t}{\text{price}_{0}}$
# 
# This will require transforming the data in the stocks object, which I do next.

# <codecell>

# df.apply(arg) will apply the function arg to each column in df, and return a DataFrame with the result
# Recall that lambda x is an anonymous function accepting parameter x; in this case, x will be a pandas Series object
stock_return = stocks.apply(lambda x: x / x[0])
stock_return.head()

# <codecell>

stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)

# <markdowncell>

# This is a much more useful plot. We can now see how profitable each stock was since the beginning of the period. Furthermore, we see that these stocks are highly correlated; they generally move in the same direction, a fact that was difficult to see in the other charts.
# 
# Alternatively, we could plot the change of each stock per day. One way to do so would be to plot the percentage increase of a stock when comparing day $t$ to day $t + 1$, with the formula:
# 
# $\displaystyle\text{growth}_t = \dfrac{\text{price}_{t+1}-\text{price}_t}{\text{price}_t}$
# 
# But change could be thought of differently as:
# 
# $\displaystyle\text{increase}_t = \dfrac{\text{price}_t-\text{price}_{t-1}}{\text{price}_t}$
# 
# These formulas are not the same and can lead to differing conclusions, but there is another way to model the growth of a stock: with log differences.
# 
# $\displaystyle\text{change}_t = \log{\left(\text{price}_t\right)} - \log{\left(\text{price}_{t-1}\right)}$
# 
# (Here, $\log$ is the natural log, and our definition does not depend as strongly on whether we use $\log(\text{price}_{t}) - \log(\text{price}_{t - 1})$ or $\log(\text{price}_{t+1}) - \log(\text{price}_{t})$.) The advantage of using log differences is that this difference can be interpreted as the percentage change in a stock but does not depend on the denominator of a fraction.
# 
# We can obtain and plot the log differences of the data in stocks as follows:

# <codecell>

# Let's use NumPy's log function, though math's log function would work just as well
import numpy as np
 
stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()

# <codecell>

stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)

# <markdowncell>

# Which transformation do you prefer? Looking at returns since the beginning of the period make the overall trend of the securities in question much more apparent. Changes between days, though, are what more advanced methods actually consider when modelling the behavior of a stock. so they should not be ignored.

# <markdowncell>

# ## Moving Averages
# Charts are very useful. In fact, some traders base their strategies almost entirely off charts (these are the “technicians”, since trading strategies based off finding patterns in charts is a part of the trading doctrine known as technical analysis). Let’s now consider how we can find trends in stocks.
# 
# Moving averages come in various forms, but their underlying purpose remains the same: to help technical traders **track the trends of financial assets by smoothing out the day-to-day price fluctuations, or noise**.
# 
# Once determined, the resulting average is then plotted onto a chart in order to allow traders to look at smoothed data rather than focusing on the day-to-day price fluctuations that are inherent in all financial markets.

# <markdowncell>

# ### Simple Moving Average (SMA)
# - taking the arithmetic mean of a given set of values, e.g. the stock prices of the last 10 days.
# - the moving average is updated when a new value comes in.
# - many individuals argue that the usefulness of the SMA is limited because each point in the data series has the same weighting
# - Critics argue that the most recent data is more significant than the older data and should have a greater influence on the final result
# 
# ### Exponential Moving Average (EMA)
# The *exponential moving average* is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information. Learning the somewhat complicated equation for calculating an EMA may be unnecessary for many traders, since nearly all charting packages do the calculations for you. However, for you math geeks out there, here is the EMA equation:
# 
# $\displaystyle\text{EMA}_t = \left(x*\alpha\right) + \left(\text{EMA}_{t-1} * (1-\alpha)\right)$,
# 
# with $x$ being the current value of the quantity in question, e.g. price, $\alpha$ as the smoothing factor, e.g. $\alpha=\frac{2}{1+N}$, and $N$ being the number of time periods.
# 
# As an initial value for EMA, the simple moving average can be computed.
# 
# **NOTE**: EMA responds more quickly to the changing prices than SMA.
# 
# - The most common time periods used in moving averages are 15, 20, 30, 50, 100 and 200 days.
# - The shorter the time span used to create the average, the more sensitive it will be to price changes.
# - The longer the time span, the less sensitive, or more smoothed out, the average will be.
# - There is no *right* time frame to use when setting up your moving averages. The best way to figure out which one works best for you is to experiment with a number of different time periods until you find one that fits your strategy.
# 
# ### Trends
# Moving Average is used to identify trends
# - uptrend when current price is above a moving average and the average is sloping upward.
# - downtrend when current price is below a moving average and the average is sloping downward.
# 
# ### Usefulness, Limitations of Moving Averages
# - Some of the primary roles of a moving average include identifying trends and reversals, measuring the strength of an asset's momentum and determining potential areas where an asset will find support or resistance.
# - A moving average can be a great risk management tool because of its ability to identify strategic areas to stop losses.
# - Using moving averages can be very ineffective during periods where the asset is trending sideways.
# - There are many different strategies involving moving averages. The most popular is the moving average crossover.
# - Moving averages are used in the creation of a number of other very popular technical indicators such as the moving average convergence divergence (MACD) or Bollinger Bands.
# - Moving averages won't solve all your investing problems. However, when used judiciously, they can be valuable tools in planning your trading strategy.

# <markdowncell>

# `pandas` provides functionality for easily computing moving averages. I demonstrate its use by creating a 20-day (one month) moving average for the Apple data, and plotting it alongside the stock.

# <codecell>

len(apple.index)
len(apple.columns)
num_cols = len(apple.columns)
num_cols
apple.columns

# <codecell>

# compute moving averages:
#if (not 'Age' in df_pf.columns):
j = 0
movingavg = [5, 10, 15, 20, 30, 50, 100, 200]
for stock in [apple, microsoft, google, amazon]:
    #print("stock: ",stock)
    for i in movingavg:
        #stock = stock.insert(loc=num_cols+j, column=str(i)+'d', value=np.round(stock["Adj. Close"].rolling(window = i, center = False).mean(), 2))
        #stock.loc[:,str(i)+"d"] = np.round(stock["Close"].rolling(window = i, center = False).mean(), 2)
        #stock.loc[:,str(i)+"d"] = np.round(stock["Close"].rolling(window = i, center = False).mean(), 2)
        #stock = stock.assign(i=np.round(stock["Close"].rolling(window = i, center = False).mean(), 2))
        #[str(i)+"d"] = 
        #stock[str(i)+"d"] = np.round(stock["Close"].rolling(window = i, center = False).mean(), 2)
        stock.loc[:][str(i)+'d'] = np.round(stock['Adj. Close'].rolling(window = i, center = False).mean(), 2)
        j = j+1

# <codecell>

apple.head()

# <codecell>

# plot data and the moving averages:
stocknames = ['apple', 'microsoft', 'google', 'amazon']
j=0
for stock in [apple, microsoft, google, amazon]:
    plt.figure()
    stock['Close'].plot()
    for i in movingavg:
        stock[str(i)+"d"].plot()
    plt.legend()
    plt.title(stocknames[j])
    j += 1

# <markdowncell>

# - When different moving averages cross, they indicate a change in trends.
# - These crossings are what we can use as **trading signals**
# 
# ### Crossovers
# - when the price of an asset moves from one side of a moving average and closes on the other side
# - Indication of shift of momentum, can be used as basic entry or exit strategy
#  - A cross below a moving average can signal the beginning of a downtrend and would be used by traders as a signal to close out
#  - A cross above a moving average from below may suggest the beginning of a new uptrend.
# - Short term average crosses through a long-term average
#  - Identification that momentum is shifting in one direction, and a strong move is likely approaching.
# - Moving Average Ribbon:
#  - 5 day average crosses up through others: buy signal
#    - waiting for 10 day average to cross above the 20 day average is often used as confirmation
#  - increases the strength of a trend and likelihood that the trend will continue
#  - placing a ribbon (many averages) increases the confidence in the analysis.
#    

# <codecell>

stocknames = ['Apple', 'Microsoft', 'Google', 'Amazon']
j=0
timeframe = 50
for stock in [apple, microsoft, google, amazon]:
    plt.figure()
    stock['Close'][-timeframe:].plot(linewidth=4.0)
    for i in movingavg:
        stock[str(i)+"d"][-timeframe:].plot(linestyle='-', marker='o')
    plt.legend(loc='lower right', ncol=3)
    plt.title(stocknames[j])
    j += 1

# <markdowncell>

# # Stock Market Prediction with Random Forests
# https://www.quantinsti.com/blog/use-decision-trees-machine-learning-predict-stock-movements


# <codecell>

from tools.randomforests_tools import *
from sklearn.ensemble import RandomForestRegressor

# <codecell>

# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016,1,1)
end = datetime.date.today()

apple = apple_orig.copy(deep=True)
microsoft = microsoft_orig.copy(deep=True)
google = google_orig.copy(deep=True)
amazon = amazon_orig.copy(deep=True)

# cutting dates:
apple = apple.loc[start:end]
microsoft = microsoft.loc[start:end]
google = google.loc[start:end]
amazon = amazon.loc[start:end]

apple.head()

# <codecell>

forecast_out = 60
apple.loc[:,'Prediction'] = apple['Adj. Close'].shift(-forecast_out).values
X = np.array(apple.drop(['Prediction'], 1))
X = preprocessing.scale(X)
# set X_forecast equal to last 30
X_forecast = X[-forecast_out:]
# remove last 30 from X
X = X[:-forecast_out]
# price
y = np.array(apple['Prediction'])
y = y[:-forecast_out]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# <codecell>

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10000)
# Train the model on training data
rf.fit(X_train, y_train);

# <codecell>

# Testing
# Use the forest's predict method on the test data
forecast_prediction = rf.predict(X_test)

# Calculate the R2
from tools.mystatstools import rSquared
r2 = rSquared(y_test, forecast_prediction)
# Print out the mean absolute error (mae)
print('R2:', round(r2, 3))

# <codecell>

# now use the model to predict the entire X population:
forecast_prediction = rf.predict(X)

# plot real data and precition
plt.figure(figsize=(15, 10))
plt.plot(apple.index.values[-len(forecast_prediction):], y, label='Adj. Close')
plt.plot(apple.index.values[-len(forecast_prediction):], forecast_prediction, label='prediction')
plt.legend()

# <codecell>

# now use the model to predict X_forecast
forecast_prediction = rf.predict(X_forecast)

# plot real data and precition
plt.figure(figsize=(15, 10))
plt.plot(apple.index.values[-len(forecast_prediction):], apple['Adj. Close'][-forecast_out:], label='Adj. Close')
plt.plot(apple.index.values[-len(forecast_prediction):], forecast_prediction, label='prediction')
plt.legend()
