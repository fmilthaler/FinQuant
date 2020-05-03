<p align="center">
  <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/images/finquant-logo.png" width="45%">
</p>

<p align="center">
  <a href="https://GitHub.com/fmilthaler/FinQuant/stargazers/">
    <img src="https://img.shields.io/github/stars/fmilthaler/FinQuant.svg?style=social&label=Star" alt='pypi'>
  </a>
  <a href="https://pypi.org/project/FinQuant">
    <img src="https://img.shields.io/badge/pypi-v0.2.2-brightgreen.svg?style=popout" alt='pypi'>
  </a>
  <a href="https://travis-ci.org/fmilthaler/FinQuant">
    <img src="https://travis-ci.org/fmilthaler/FinQuant.svg?style=popout?branch=master" alt='travis'>
  </a>
  <a href="http://finquant.readthedocs.io/">
    <img src="https://img.shields.io/readthedocs/finquant.svg?style=popout" alt="docs">
  </a>
  <a href="https://GitHub.com/fmilthaler/FinQuant/graphs/contributors/">
    <img src="https://img.shields.io/github/contributors/fmilthaler/FinQuant.svg?style=popout" alt="contributors">
  </a>
  <a href="https://github.com/fmilthaler/FinQuant/issues">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=popout" alt="contributions">
  </a>
  <a href="https://github.com/fmilthaler/FinQuant/blob/master/LICENSE.txt">
    <img src="https://img.shields.io/github/license/fmilthaler/FinQuant.svg?style=popout" alt="license">
  </a>
</p>

# FinQuant
*FinQuant* is a program for financial **portfolio management, analysis and optimisation**.

This README only gives a brief overview of *FinQuant*. The interested reader should refer to its [documentation](https://finquant.readthedocs.io "FinQuant Documentation").

## Table of contents
 - [Motivation](#Motivation)
 - [Installation](#Installation)
 - [Portfolio Management](#Portfolio-Management)
 - [Returns](#Returns)
 - [Moving Averages](#Moving-Averages)
 - [Portfolio Optimisation](#Portfolio-Optimisation)
   - [Efficient Frontier](#Efficient-Frontier)
   - [Monte Carlo](#Monte-Carlo)
 - [Examples](#Examples)
   - [Building a portfolio with data from web](#Building-a-portfolio-with-data-from-web)
   - [Building a portfolio with preset data](#Building-a-portfolio-with-preset-data)
   - [Analysis of a portfolio](#Analysis-of-a-portfolio)
   - [Optimisation of a portfolio](#Optimisation-of-a-portfolio)

## Motivation
Within a few lines of code, *FinQuant* can generate an object that holds your stock prices of your desired financial portfolio, analyses it, and can create plots of different kinds of *Returns*, *Moving Averages*, *Moving Average Bands with buy/sell signals*, and *Bollinger Bands*. It also allows for the optimisation based on the *Efficient Frontier* or a *Monte Carlo* run of the financial portfolio within a few lines of code. Some of the results are shown here.

### Automatically generating an instance of `Portfolio`
`finquant.portfolio.build_portfolio` is a function that eases the creating of your portfolio. See below for one of several ways of using `build_portfolio`.
```
from finquant.portfolio import build_portfolio
names = ['GOOG', 'AMZN', 'MCD', 'DIS']
start_date = '2015-01-01'
end_date = '2017-12-31'
pf = build_portfolio(names=names,
                    start_date=start_date,
                    end_date=end_date)
```
`pf` is an instance of `finquant.portfolio.Portfolio`, which contains the prices of the stocks in your portfolio. Then...
```
pf.data.head(3)
```
yields
```
              GOOG    AMZN        MCD        DIS
Date
2015-01-02  524.81  308.52  85.783317  90.586146
2015-01-05  513.87  302.19  84.835892  89.262380
2015-01-06  501.96  295.29  84.992263  88.788916
```

### Portfolio properties
Nicely printing out the portfolio's properties
```
pf.properties()
```
Depending on the stocks within your portfolio, the output looks something like the below.
```
----------------------------------------------------------------------
Stocks: GOOG, AMZN, MCD, DIS
Time window/frequency: 252
Risk free rate: 0.005
Portfolio expected return: 0.266
Portfolio volatility: 0.156
Portfolio Sharpe ratio: 1.674

Skewness:
       GOOG      AMZN      MCD       DIS
0  0.124184  0.087516  0.58698  0.040569

Kurtosis:
       GOOG      AMZN       MCD       DIS
0 -0.751818 -0.856101 -0.602008 -0.892666

Information:
   Allocation  Name
0        0.25  GOOG
1        0.25  AMZN
2        0.25   MCD
3        0.25   DIS
----------------------------------------------------------------------
```

### Cumulative Return
```
pf.comp_cumulative_returns().plot().axhline(y = 0, color = "black", lw = 3)
```
yields
<p align="center">
  <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/images/cumulative-return.svg?sanitize=true" width="60%">
</p>

### Band Moving Average (Buy/Sell Signals)
```
from finquant.moving_average import compute_ma, ema
# get stock data for disney
dis = pf.get_stock("DIS").data.copy(deep=True)
spans = [10, 50, 100, 150, 200]
ma = compute_ma(dis, ema, spans, plot=True)
```
yields
<p align="center">
  <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/images/ma-band-buysell-signals.svg?sanitize=true" width="60%">
</p>

### Bollinger Band
```
from finquant.moving_average import plot_bollinger_band
# get stock data for disney
dis = pf.get_stock("DIS").data.copy(deep=True)
span=20
plot_bollinger_band(dis, sma, span)
```
yields
<p align="center">
  <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/images/bollinger-band.svg?sanitize=true" width="60%">
</p>

### Portfolio Optimisation
```
# performs and plots results of Monte Carlo run (5000 iterations)
opt_w, opt_res = pf.mc_optimisation(num_trials=5000)
# plots the results of the Monte Carlo optimisation
pf.mc_plot_results()
# plots the Efficient Frontier
pf.ef_plot_efrontier()
# plots optimal portfolios based on Efficient Frontier
pf.ef.plot_optimal_portfolios()
# plots individual plots of the portfolio
pf.plot_stocks()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/images/ef-mc-overlay.svg?sanitize=true" width="60%">
</p>

## Installation
As it is common for open-source projects, there are several ways to get hold of the code. Choose whichever suits you and your purposes best.

### Dependencies
*FinQuant* depends on the following Python packages:
 - python>=3.5.0
 - numpy>=1.15
 - pandas>=0.24
 - matplotlib>=1.5.1
 - quandl>=3.4.5
 - yfinance>=0.1.43
 - scipy>=1.2.0
 - pytest>=2.8.7

### From PyPI
*FinQuant* can be obtained from PyPI

```pip install FinQuant```

### From GitHub
Get the code from GitHub:

```git clone https://github.com/fmilthaler/FinQuant.git```

Then inside `FinQuant` run:

```python setup.py install```

Alternatively, if you do not wish to install *FinQuant*, you can also download/clone it as stated above, and then make sure to add it to your ``PYTHONPATH``.

## Portfolio Management
This is the core of *FinQuant*. `finquant.portfolio.Portfolio` provides an object that holds prices of all stocks in your portfolio, and automatically computes the most common quantities for you. To make *FinQuant* an user-friendly program, that combines data analysis, visualisation and optimisation, the object provides interfaces to the main features that are provided in the modules in `./finquant/`.

To learn more about the object, please read through the [documentation](https://finquant.readthedocs.io/en/latest/ "FinQuant Documentation"), docstring of the module/class, and/or have a look at the examples.

`finquant.portfolio.Portfolio` also provides a function `build_portfolio` which is designed to automatically generate an instance of `Portfolio` for the user's convenience. For more information on how to use `build_portfolio`, please refer to the [documentation](https://finquant.readthedocs.io/en/latest/ "FinQuant Documentation"), its `docstring` and/or have a look at the examples.

## Returns
Daily returns of stocks are often computed in different ways. *FinQuant* provides three different ways of computing the daily returns in `finquant.returns`:
1. The cumulative return: <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/tex/738645698dc3073b4bb52a0c078ae829.svg?invert_in_darkmode&sanitize=true" align=middle width=194.52263655pt height=46.976899200000005pt/>
2. Percentage change of daily returns: <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/tex/27215e5f36fd0308b51ab510444edf0d.svg?invert_in_darkmode&sanitize=true" align=middle width=126.07712039999997pt height=48.84266309999997pt/>
3. Log Return: <img src="https://raw.githubusercontent.com/fmilthaler/finquant/master/tex/ef37c00ad58fe657a64041c3093e0640.svg?invert_in_darkmode&sanitize=true" align=middle width=208.3327686pt height=57.53473439999999pt/>

In addition to those, the module provides the function `historical_mean_return(data, freq=252)`, which computes the historical mean of the daily returns over a time period `freq`.

## Moving Averages
The module `finquant.moving_average` allows the computation and visualisation of Moving Averages of the stocks listed in the portfolio is also provided. It entails functions to compute and visualise the
 - `sma`: Simple Moving Average, and
 - `ema`: Exponential Moving Average.
 - `compute_ma`: a Band of Moving Averages (of different time windows/spans) including Buy/Sell signals
 - `plot_bollinger_band`: a Bollinger Band for
   - `sma`,
   - `ema`.

## Portfolio Optimisation
### Efficient Frontier
An implementation of the Efficient Frontier (`finquant.efficient_frontier.EfficientFrontier`) allows for the optimisation of the portfolio for
 - `minimum_volatility` Minimum Volatility,
 - `maximum_sharpe_ratio` Maximum Sharpe Ratio
 - `efficient_return` Minimum Volatility for a given expected return
 - `efficient_volatility` Maximum Sharpe Ratio for a given target volatility

by performing a numerical solve to minimise/maximise an objective function.

Often it is useful to visualise the *Efficient Frontier* as well as the optimal solution. This can be achieved with the following methods:
 - `plot_efrontier`: Plots the *Efficient Frontier*. If no minimum/maximum Return values are provided, the algorithm automatically chooses those limits for the *Efficient Frontier* based on the minimum/maximum Return values of all stocks within the given portfolio.
 - `plot_optimal_portfolios`: Plots markers of the portfolios with the Minimum Volatility and Maximum Sharpe Ratio.

For reasons of user-friendliness, interfaces to these functions are provided in `finquant.portfolio.Portfolio`. Please have a look at the [documentation](https://finquant.readthedocs.io "FinQuant Documentation").

### Monte Carlo
Alternatively a *Monte Carlo* run of `n` trials can be performed to find the optimal portfolios for
 - minimum volatility,
 - maximum Sharpe ratio

The approach branded as *Efficient Frontier* should be the preferred method for reasons of computational effort and accuracy. The latter approach is only included for the sake of completeness, and creation of beautiful plots.

## Examples
For more information about the project and details on how to use it, please
look at the examples provided in `./example`.

**Note**: In the below examples, `pf` refers to an instance of `finquant.portfolio.Portfolio`, the object that holds all stock prices and computes its most common quantities automatically. To make *FinQuant* a user-friendly program, that combines data analysis, visualisation and optimisation, the object also provides interfaces to the main features that are provided in the modules in `./finquant/` and are discussed throughout this README.

### Building a portfolio with data from web
`./example/Example-Build-Portfolio-from-web.py`: Shows how to use *FinQuant* to build a financial portfolio by downloading stock price data through the Python package `quandl`/`yfinance`.

### Building a portfolio with preset data
`./example/Example-Build-Portfolio-from-file.py`: Shows how to use *FinQuant* to build a financial portfolio by providing stock price data yourself, e.g. by reading data from disk/file.

### Analysis of a portfolio
`./example/Example-Analysis.py`: This example shows how to use an instance of `finquant.portfolio.Portfolio`, get the portfolio's quantities, such as
 - Expected Returns,
 - Volatility,
 - Sharpe Ratio.

It also shows how to extract individual stocks from the given portfolio. Moreover it shows how to compute and visualise:
 - the different Returns provided by the module `finquant.returns`,
 - *Moving Averages*, a band of *Moving Averages*, and a *Bollinger Band*.

### Optimisation of a portfolio
`./example/Example-Optimisation.py`: This example focusses on the optimisation of a portfolio. To achieve this, the example shows the usage of `finquant.efficient_frontier.EfficientFrontier` for optimising the portfolio, for the
 - Minimum Volatility
 - Maximum Sharpe Ratio
 - Minimum Volatility for a given target Return
 - Maximum Sharpe Ratio for a given target Volatility.

Furthermore, it is also shown how the entire *Efficient Frontier* and the optimal portfolios can be computed and visualised. If needed, it also gives an example of plotting the individual stocks of the given portfolio within the computed *Efficient Frontier*.

Also, the optimisation of a portfolio and its visualisation based on a *Monte Carlo* is shown.

Finally, *FinQuant*'s visualisation methods allow for overlays, if this is desired. Thus, with only the following few lines of code, one can create an overlay of the *Monte Carlo* run, the *Efficient Frontier*, its optimised portfolios for *Minimum Volatility* and *Maximum Sharpe Ratio*, as well as the portfolio's individual stocks.

