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

# write data to disk:
#pf.getPortfolio().to_csv("my-pf.csv", encoding='utf-8', index=False, header=True)
#pf.getPfStockData().to_csv("my-pfstockdata.csv", encoding='utf-8', index=True, index_label="Date")

# <markdowncell>

# ## Get data disk/file

# <codecell>

#test = pd.read_csv("my-pf.csv")
pf_info = pd.read_csv("../data/ex1-portfolio.csv")
pfstockdata = pd.read_csv("../data/ex1-stockdata.csv", index_col='Date', parse_dates=True)

# <codecell>

pf_info

# <codecell>

pfstockdata.head(3)

# <codecell>



# <codecell>

# build portfolio

#print(pf)
#pf.getPfStockData().head(3)
#pf.getStocks()['AMZN'].getStockData().head(3)

# <codecell>



# <codecell>



# <codecell>


