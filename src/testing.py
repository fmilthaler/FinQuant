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



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>

#for row in df_pf.itertuples(index=True, name='Pandas'):
columns = df_pf.columns
funds = {}
for idx, row in df_pf.iterrows():
    #print(row.Vintage, row.FMV)
    investmentInfo = row
    #investmentInfo = investmentInfo.rename("bla")
    #print(investmentInfo.name)
    #print("FMV = ", investmentInfo.FMV)
    #print(row.Region)
    fund = portfolio.Fund(investmentInfo, [])
    funds.update({fund.name : fund})
    #print(fund.investmentinfo)
columns
print(funds["Fund3"])
#funds["Fund3"].investmentinfoFMV

# build portfolio
pf = portfolio.Portfolio2("my Portfolio")
pf.addFund(funds["Fund0"])
pf.addFund(funds["Fund1"])
pf.addFund(funds["Fund2"])

#print(pf)
#print(pf.funds)
#print(funds["Fund0"])
#pf.getPortfolio()
#print(pf.funds["Fund0"])
print("+++++++++++")
print(funds['Fund0'].investmentinfo.values)
#print(pf)
#for key, value in funds.items():
#    print(key+": "+str(value))


print("+++++++++++")
print(pf.getPortfolio())
print(pf.getPortfolio().loc[0])

#pf.getPortfolio().query('Fund == Fund2')

#print(pf.getFunds()["Fund0"].investmentinfo.values)
#print(funds)
#pd.DataFrame.from_dict(funds, orient='index', columns=["ID","Fund","Vintage","Region","Strategy","CCY","FMV"])

print("-----------")
print(pf)
print("-----------")
#pf.getPortfolio()[pf.getPortfolio().Fund == "Fund2"]
#pf.getPortfolio().loc[pf.getPortfolio()['Fund'] == 'Fund2']

# <codecell>

# build portfolio object
pf = portfolio.Portfolio("my Portfolio",df_pf, df_data)

# print portfolio information
print(pf)
print("-------------------")
pf.getFund("Fund2")

# <codecell>

df_data.tail()

# <codecell>

# for cross reference: get reference date,
# compute the age of the funds and add it (age)
# to the portfolio dataframe:
filename = "../data/portfolio.csv"
refDate = mt.getReferenceDate(filename)
refYear = mt.getReferenceYear(refDate)

if (not 'Age' in df_pf.columns):
    df_pf.insert(loc=3, column='Age',
                 value=refYear - df_pf['Vintage'].values)

# vintage in the data dataframe
if (not 'Vintage' in df_data.columns):
    df_data.insert(loc=2, column='Vintage',
                   value=refYear-df_data['Age'])

# <codecell>

# pivot the dataframe:
df_roi = pd.pivot_table(df_data, index=['Age', 'Vintage'],
                     columns=['Strategy'],
                     values=['ROI'],
                     aggfunc={'ROI': 
                              [np.mean, np.std, np.min, np.max, np.var]},
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
# saving figure in subfolder images
imgname = "exercise1-barplot-avROIs-age-bothStrats.pdf"
fig.savefig(os.path.join('images', imgname),
            bbox_inches='tight', pad_inches=0.1)

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

# average ROI per vintage
df_vintage = pd.pivot_table(df_data, index=['Vintage'],
                     values=['ROI'],
                     aggfunc={'ROI': 
                              [np.mean, np.std, np.min, np.max, np.var]},
                              #[np.mean, np.var]},
                     fill_value = 0
                    )
df_vintage

# <codecell>

# bar plot in seaborn:
fig=pylab.figure()
vintages = df_vintage.index.values
ax = sns.barplot(x="Vintage", y="ROI", ci=False, data=df_data)
pylab.xlabel('Vintage')
pylab.ylabel('Average ROI')
pylab.title('Average ROIs plotted over vintages')
mt.addValToBarPlot(ax, df_vintage['ROI']['mean'].values,
                   xshift=0.15, yshift=-.1)
pylab.show()
# saving figure in subfolder images
imgname = "exercise1-barplot-avROIs-accumulatedStrats.pdf"
fig.savefig(os.path.join('images', imgname),
            bbox_inches='tight', pad_inches=0.1)

# <codecell>



# <codecell>


