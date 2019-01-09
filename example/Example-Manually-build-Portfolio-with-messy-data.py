# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # This example shows how to build a portfolio by manually adding return data
# 
# Note: The stock/fund data is provided in two data files. The data itself is not a real world example but simply a set of somewhat random numbers that were generated and are used as a showcase here.

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd

# <codecell>

# importing QPY's portfolio and stock
from qpy.portfolio import Portfolio, Stock

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

# <markdowncell>

# ## Getting data from files
# Here we use pandas `read_csv` function to read in two files:
#  * `../data/ex2-portfolio.csv`: contains information about the portfolio, e.g. Name of the funds, currency, fair market value (FMV) and more
#  * `../data/ex2-data.csv`: this is an example of a messy datafile which contains return of investment information of all funds in one column. The return data can be extracted manually by matching data provided in this file and `../data/ex2-portfolio.csv`, e.g. columns `Year` and `Age`. How this can be done is shown below.

# <codecell>

# read data into pandas dataframe:
df_pf_orig = pd.read_csv("../data/ex2-portfolio.csv")
df_data_orig = pd.read_csv("../data/ex2-data.csv")
# make copies
df_pf = df_pf_orig.copy(deep=True)
df_data = df_data_orig.copy(deep=True)
# dropping redundant columns
df_pf.drop(columns=['ID'],inplace=True)
df_data.drop(columns=['Index'],inplace=True)

# <codecell>

df_pf

# <codecell>

df_data.head(3)

# <markdowncell>

# ## Adding Age of the funds to `df_pf`
# As mentioned above, this data is messy. To better match the return data in `df_data` to the portfolio information in `df_pf`, we manually add a column listing the age of each fund to `df_pf`.
# 
# In order to do this, a reference year is required. Here we assume the reference year is 2018.

# <codecell>

refYear = 2018
if (not 'Age' in df_pf.columns):
    df_pf.insert(loc=3, column='Age',
                 value=refYear - df_pf['Year'].values)
df_pf

# <markdowncell>

# ## Extract return data from `df_data` (pandas.DataFrame) and feed it into the Portfolio data structure
# Below a short function is written that extract data from the DataFrame `df` based on a given age and strategy.

# <codecell>

# custom function to deal with the messy data given in the data file
def extractRoiData(df, age, strategy):
    # generate a string to query the relevant data:
    querystring = 'Age=='+str(age)+' & Strategy=="'+str(strategy)+'"'
    # get data for the given information:
    roi_data = df.query(querystring).reset_index(drop=True)['ROI']
    return pd.DataFrame(roi_data)

# <markdowncell>

# ## Building a portfolio
# To create a portfolio with QPY, we start off by creating an empty instance of Portfolio.

# <codecell>

# build portfolio object
# first: creating an empty portfolio
pf = Portfolio()

# <markdowncell>

# Then we loop over each fund listed in `df_pf`, create an instance of `Stock` and pass it the relevant stock information in `df_pf` as well as the return (ROI) data in `df_data`.
# 
# To add a stock/fund manually to a portfolio, the method `portfolio.addStock(<Stock>)` is used.


# <codecell>

for i in range(len(df_pf)):
    #print(df_pf.loc[i])
    age = df_pf.loc[i].Age
    strategy = df_pf.loc[i].Strategy
    #data = extractRoiData()
    pf.addStock(Stock(df_pf.loc[i], extractRoiData(df_data, age, strategy)))

# <markdowncell>

# ## At this point, the portfolio is successfully built.
# Out of the messy data in the data files, we obtained a structured object that contains all the data within a few steps.
# 
# The data can be used for further analysis and computation, some of which is shown in other examples. Below we just briefly look at the obtained object.

# <codecell>

# the object "pf" is the obtained portfolio
pf.getPortfolio()

# <codecell>

pf.getPfRoiData().head()

# <codecell>

pf.getPfRoiData().describe()

# <markdowncell>

# ## Skew and Kurtosis for each stock individually
# The skewness and Kurtosis of each stock/fund can be looked at individually as shown below.
# 
# In order to do this, we get all `Stock` instances within `pf` with `pf.getStocks()`. With `Stock.compSkew()` and `Stock.compKurtosis()` the skewness and Kurtosis are computed.

# <codecell>

# skew and kurtosis of each stock:
for label, stock in pf.getStocks().items():
    print("++++++++++++++++++")
    print(str(label)+":")
    print("Skewness: {0:0.2f}".format(stock.compSkew()))
    print("Kurtosis: {0:0.2f}".format(stock.compKurtosis()))
    print(stock)

# <markdowncell>

# ## Skew and Kurtosis can also be done on the entire portfolio
# The same can be achieved on a portfolio level. The commands are `pf.compSkew()` and `pf.compKurtosis()`.

# <codecell>

print("Fund skewness:\n{}\n".format(pf.compSkew()))
print("Fund Kurtosis:\n{}".format(pf.compKurtosis()))

# <markdowncell>

# # Computing the expected ROI and volatility of a portfolio
# The expected return and volatility of a portfolio are computed while the portfolio is being built. The user can get those quantities as follows:
#  * `pf.getPfExpectedRoi()`
#  * `pf.getPfVolatility()`

# <codecell>

exp_roi = pf.getPfExpectedRoi()
volatility = pf.getPfVolatility()
print("Expected return = {0:0.2f}".format(exp_roi))
print("Volatility = {0:0.2f}".format(volatility))

# <markdowncell>

# # Optimisation of portfolio
# 
# The given portfolio can be optimised for
#  * Minimum volatility, and
#  * Maximum Sharpe ratio.
# 
# This is done by performing a Monte Carlo simulation that randomly selects the weights of the portfolio and then computes the Sharpe ratio. An example is shown below.
# 
# More information on the input arguments of this method can be seen with `print(pf.optimisePortfolio.__doc__)`

# <codecell>

print(pf.optimisePortfolio.__doc__)

# <codecell>

num_trials = 25000
riskfreerate = 0.005
# the data provided in the data files is not real world data, hence
# the return values are rather extreme. Hence, the period is set to 1
# rather than 252.
pf_opt = pf.optimisePortfolio(num_trials=num_trials,
                              riskfreerate=riskfreerate,
                              period=1,
                              plot=True)

# <markdowncell>

# ### Examining the optimised portfolios
# The optimised portfolios are stored in `pf_opt`.

# <codecell>

pf_opt

# <markdowncell>

# ## Printing out the optimised Portfolio

# <codecell>

# print out optimised portfolio information:
print("The portfolio with the highest Sharpe ratio is:")
print(pf_opt.loc['Max Sharpe Ratio'].to_frame().T)

print("\nAnd the portfolio with the minimum volatility is:")
print(pf_opt.loc['Min Volatility'].to_frame().T)

# <codecell>

# sanity check: comparing sum of FMV of initial and optimised portfolios:
labels = ['Max Sharpe Ratio', 'Min Volatility']
for label in labels:
    total = 0
    for i in range(6):
        total += pf_opt.loc['Min Volatility']['Fund'+str(i)]
    print("Sum of FMV of {0} portfolio = {1:.5f}".format(label, total))
print("Sum of FMV of {0} = {1:.5f}".format('Initial Portfolio', pf.getPortfolio().FMV.sum()))

# <markdowncell>

# ## Plotting Return data
# First by extracting one stock/fund from `pf` and plotting its data, and further below by extracting the data of all stocks/funds from `pf`.

# <codecell>

pf.getPfRoiData()['Fund0'].plot()
pf.getPfRoiData().plot()
