# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# ## Analysis of a financial portfolio

# <codecell>

#from pylab import *
import pylab
import matplotlib.pyplot as plt
import math, random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# plotting style:
sns.set_style('darkgrid')

# <codecell>

# importing some custom functions
from qpy.portfolio import Portfolio
from qpy.fund import Fund
#from portfolio import expectedValue, volatility

# <codecell>

import tools.mytools as mt

# <codecell>

# to plot within the notebook:
#%pylab inline
#pylab

# <codecell>

#import random, pylab, numpy

##set line width
pylab.rcParams['lines.linewidth'] = 2
##set font size for titles 
pylab.rcParams['axes.titlesize'] = 14
##set font size for labels on axes
pylab.rcParams['axes.labelsize'] = 12
##set size of numbers on x-axis
pylab.rcParams['xtick.labelsize'] = 10
##set size of numbers on y-axis
pylab.rcParams['ytick.labelsize'] = 10

# <markdowncell>

# ## Getting data from files

# <codecell>

# read data into pandas dataframe:
#df_pf_orig = pd.read_csv("../data/portfolio.csv", skiprows=1)
df_pf_orig = pd.read_csv("../data/ex1-portfolio.csv")
df_data_orig = pd.read_csv("../data/ex1-data.csv")#, usecols=[0,1,2,3,4])
# make copies
df_pf = df_pf_orig.copy(deep=True)
df_data = df_data_orig.copy(deep=True)
# dropping redundant column
df_pf.drop(columns=['ID'],inplace=True)
df_data.drop(columns=['Index'],inplace=True)

# <codecell>

df_data.head()

# <codecell>

df_pf

# <markdowncell>

# ## Adding a column to dataframe

# <codecell>

refYear = 2018
if (not 'Age' in df_pf.columns):
    df_pf.insert(loc=3, column='Age',
                 value=refYear - df_pf['Year'].values)
df_pf

# <markdowncell>

# ## Extract data from pandas.DataFrame and feed it into the Portfolio data structure

# <codecell>

def extractRoiData(df, age, strategy):
    # generate a string to query the relevant data:
    querystring = 'Age=='+str(age)+' & Strategy=="'+str(strategy)+'"'
    # get data for the given information:
    roi_data = df.query(querystring).reset_index(drop=True).ROI
    return pd.DataFrame(roi_data)

# <codecell>

# build portfolio object
ref_year = 2018

pf = Portfolio('my Portfolio', ref_year)
for i in range(len(df_pf)):
    #print(df_pf.loc[i])
    age = df_pf.loc[i].Age
    strategy = df_pf.loc[i].Strategy
    #data = extractRoiData()
    pf.addFund(Fund(df_pf.loc[i], extractRoiData(df_data, age, strategy)))


# <markdowncell>

# ## At this point, the portfolio data structure is completed
# The data can be examined as below

# <codecell>

pf.getPortfolio()

# <codecell>

fund0 = pf.getFund('Fund0')
fund0.getRoiData().describe()

# <codecell>

pf.getPfRoiData().head()

# <codecell>

pf.getPfRoiData().describe()

# <markdowncell>

# ## Skew and Kurtosis for each fund individually

# <codecell>

# skew and kurtosis of each fund:
for label, fund in pf.getFunds().items():
    print("++++++++++++++++++")
    print(str(label)+":")
    print("Skew: %.2f" % fund.compSkew())
    print("Kurtosis: %.2f" % fund.compKurtosis())

# <markdowncell>

# ## Skew and Kurtosis can also be done on the entire portfolio

# <codecell>

pf.compSkew()

# <codecell>

pf.compKurtosis()

# <markdowncell>

# # Computing the expected ROI and volatility of a portfolio
# These are done as shown below.

# <codecell>

exp_roi = pf.compPfExpectedRoi()
volatility = pf.compPfVolatility()
print("expected ROI = %.2f" % exp_roi)
print("volatility = %.2f" % volatility)

# <markdowncell>

# # Optimisation of portfolio
# This is done by performing a Monte Carlo simulation that randomly selects the weights of the portfolio and then computes the Sharpe ratio. An example is shown below.

# <codecell>

num_trials = 10000
riskfreerate = 0
(max_sharpe_port, min_vol_port) = pf.optimisePortfolio(num_trials, riskfreerate=riskfreerate, plot=True)

# <markdowncell>

# ## The optimised Portfolio

# <codecell>

print("The portfolio with the highest Sharpe ratio is:")
print(max_sharpe_port)
print("\nAnd the portfolio with the minimum volatility is:")
print(min_vol_port)

# <codecell>

pd.DataFrame([max_sharpe_port])

# <codecell>

pd.DataFrame([min_vol_port])

# <codecell>



# <codecell>



# <codecell>



# <codecell>

sns.pairplot(pf.getPfRoiData())

# <codecell>

sns.kdeplot(pf.getPfRoiData(), shade=True)
