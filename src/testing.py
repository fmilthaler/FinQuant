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
import portfolio

# <codecell>

import mytools as mt

# <codecell>

# to plot within the notebook:
%pylab inline
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

# <codecell>

# read data into pandas dataframe:
#df_pf_orig = pd.read_csv("../data/portfolio.csv", skiprows=1)
df_pf_orig = pd.read_csv("../data/portfolio.csv")
df_data_orig = pd.read_csv("../data/data.csv")#, usecols=[0,1,2,3,4])
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

# <codecell>

# build portfolio object
pf = portfolio.Portfolio("my Portfolio", 2018, df_pf, df_data)

# print portfolio information
print(pf)
print("-------------------")
pf.getFund("Fund2")

# <codecell>

#pf.pf_roi_data
pf.getPfRoiData().describe()

# <codecell>



# <codecell>

def SharpeRatio(x):
    return np.sqrt(len(x)) * x.mean() / x.std()


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

# pivot the dataframe:
#df_roi = pd.pivot_table(pf.getPfRoiData(), index=['Age', 'Strategy'],
df_roi = pd.pivot_table(df_data, index=['Age'],
                     columns=['Strategy'],
                     values=['ROI'],
                     aggfunc={'ROI': 
                              [np.mean, np.std, np.min, np.max, np.var, SharpeRatio]},
                              #[np.mean, np.std, np.min, np.max, np.var]},
                              #[np.mean, np.var]},
                     fill_value = 0
                    )
df_roi

# <codecell>

# Plotting means of ROI
# bar plot in seaborn:
fig=pylab.figure(figsize=(9,5))
ax = sns.barplot(x="Age", y="ROI", hue="Strategy", ci=None, data=df_data)
pylab.ylim([0,1.75])
pylab.xlabel('Age')
pylab.ylabel('Average ROI')
pylab.title('Average ROIs plotted over ages for different strategies')
# adding values to bars:
buyout_mean = df_roi['ROI']['mean']['Buyout'].values
vc_mean = df_roi['ROI']['mean']['VC'].values
all_means = list(np.array([buyout_mean, vc_mean]).flatten())
mt.changeBarWidth(ax,.4)
mt.addValToBarPlot(ax, all_means, xshift=0.075, precision=2)
pylab.show()

# <codecell>

# Comparison of mean ROI obtained  different strategies
strategies = ("Buyout", "VC")
# first computing overall mean of ROI:
roi_means = {}
for strategy in strategies:
    roi_means[strategy] = df_data.query('Strategy=="'+strategy+'"')['ROI'].mean()
    print("Strategy: "+strategy)
    print("Overall mean of ROI =",
          round(roi_means[strategy],6))

# Print out of result:
key_max = max(roi_means.keys(), key=(lambda k: roi_means[k]))
print("The maximum mean ROI of "
      +str(round(roi_means[key_max], 5))
      +" is obtained by strategy "+key_max+".")

# <codecell>

# extra insights:
# highest (max) mean of ROI per strategy per year:
print("The maximum mean ROI of "
      +str(round(df_roi['ROI']['mean'].max().max(),5))
      +" is obtained by the strategy "
      +str(df_roi['ROI']['mean'].max().idxmax())+".")

print("However, the maximum ROI of "
     +str(round(df_data.max()['ROI'],5))
     +" is obtained by the strategy "+str(df_data.max()['Strategy'])
     +" with an age of 6 (Vintage: 2011)")

# <codecell>



# <codecell>


