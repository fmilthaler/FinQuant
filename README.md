# QuantPy
`QuantPy` is a program for financial portfolio management. It combines various stocks or funds to form a portfolio. Most common quantities, such as expected annual return, volatility, and Sharpe Ratio are computed as the portfolio object is being created and can be obtained easily.

Daily returns of stocks are often computed in different ways. `QuantPy` provides three different ways of computing the daily returns:
1. <img src="/tex/9af736057521ad784ede96a7bafcdcbf.svg?invert_in_darkmode&sanitize=true" align=middle width=105.11441654999997pt height=24.65753399999998pt/>
2. Percentage change of daily returns
3. Log Return: <img src="/tex/aa4fdd53b3fb5a2512e7cb9f674b6c43.svg?invert_in_darkmode&sanitize=true" align=middle width=309.58945215pt height=24.65753399999998pt/>

The module `quantpy.moving_average` allows the computation and visualisation of Moving Averages of the stocks listed in the portfolio is also provided. It entails functions to compute and visualise the
 - Simple Moving Average (SMA), and
 - Exponential Moving Average (EMA).
 - a Band of Moving Averages (of different time windows/spans) including Buy/Sell signals
 - a Bollinger Band for
   - SMA, and
   - EMA.

Moreover, an implementation of the Efficient Frontier allows for the optimisation of the portfolio for
 - minimum volatility,
 - maximum Sharpe ratio
 - minimum volatility for a given expected return
 - maximum Sharpe ratio for a given target volatility

by performing a numerical solve  to minimise/maximise an objective function.

Alternatively a *Monte Carlo* run of `n` trials can be performed to find the optimal portfolios for
 - minimum volatility,
 - maximum Sharpe ratio

The approach branded as *Efficient Frontier* should be the preferred method for reasons of computational effort and accuracy. The latter approach is only included for the sake of completeness, and creation of beautiful plots.


and  allows for an optimisation of the given portfolio.

For more information about the project and details on how to use it, please
look at the examples provided in `./example`.
