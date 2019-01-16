'''
Provides optimisation functions
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from qpy.returns import dailyReturns
from qpy.quants import annualised_portfolio_quantities


def random_portfolios(data, num_trials, riskFreeRate=0.005, freq=252):
    '''
    Generates and returns a number of random weights/portfolios
    (sum of weights = 1), computes their expected annual return,
    volatility and Sharpe ratio.

    Input:
     * data: A DataFrame which contains the stock data (e.g. prices) of
         relevant stocks. Note, there should only be one data column
         per stock, or else this will fail.
     * num_trials: Integer (default: 10000), number of portfolios to be
         computed, each with a random distribution of weights/investments
         in each stock
     * riskFreeRate: Float (default: 0.005), the risk free rate as
         required for the Sharpe Ratio
     * freq: Integer (default: 252), number of trading days, default
         value corresponds to trading days in a year

    Output:
     * df_weights: pandas.DataFrame, holds the weights for each randomly
         generated portfolio
     * df_results: pandas.DataFrame, holds expected annualised return,
         volatility and Sharpe ratio of each randomly generated portfolio
    '''
    # set number of stocks in the portfolio
    num_stocks = len(data.columns)
    # set up array to hold results
    weights_columns = list(data.columns)
    result_columns = ['Expected Return', 'Volatility', 'Sharpe Ratio']
    weights = np.zeros((num_stocks, num_trials))
    results = np.zeros((len(result_columns), num_trials))
    # compute returns, means and covariance matrix
    returns = dailyReturns(data)
    return_means = returns.mean()
    cov_matrix = returns.cov()
    # Monte Carlo simulation
    for i in range(num_trials):
        # select random weights for portfolio
        w = np.array(np.random.random(num_stocks))
        # rebalance weights
        w = w/np.sum(w)
        # compute portfolio return and volatility
        portfolio_values = annualised_portfolio_quantities(
            w, return_means, cov_matrix)
        # store random weights
        weights[:, i] = w
        # store results in results array
        results[:, i] = portfolio_values
    # transpose and convert to pandas.DataFrame:
    df_weights = pd.DataFrame(weights.T, columns=weights_columns)
    df_results = pd.DataFrame(results.T, columns=result_columns)
    return df_weights, df_results


def optimiseMC(data,
               total_investment,
               num_trials=10000,
               riskFreeRate=0.005,
               freq=252,
               initial_weights=None,
               verbose=True,
               plot=True):
    '''
    Optimisation of the portfolio by performing a Monte Carlo simulation.

    Input:
     * data: A DataFrame which contains the stock data (e.g. prices) of
         relevant stocks. Note, there should only be one data column
         per stock, or else this will fail.
     * total_investment: Float, total amount of money to be invested.
     * num_trials: Integer (default: 10000), number of portfolios to be
         computed, each with a random distribution of weights/investments
         in each stock
     * riskFreeRate: Float (default: 0.005), the risk free rate as
         required for the Sharpe Ratio
     * freq: Integer (default: 252), number of trading days, default
         value corresponds to trading days in a year
     * initial_weights: List/Array (default: None), weights of
         initial/given portfolio, only used to plot a marker for the
         initial portfolio in the optimisation plot.
     * verbose: Boolean (default: True), if True, prints out optimised
         portfolio allocations
     * plot: Boolean (default: True), if True, a plot showing the results
         is produced

    Output:
     * opt: DataFrame with optimised investment strategies for maximum
         Sharpe Ratio and minimum volatility.
    '''
    if (initial_weights is not None and
       not isinstance(initial_weights, np.ndarray)):
        raise ValueError("If given, optional argument 'initial_weights' "
                         + "must be of type numpy.ndarray")

    # compute returns, means and covariance matrix
    returns = dailyReturns(data)
    return_means = returns.mean()
    cov_matrix = returns.cov()

    # computed expected return and volatility of initial portfolio
    if (initial_weights is not None):
        initial_values = annualised_portfolio_quantities(initial_weights,
                                                         return_means,
                                                         cov_matrix)
        initial_return = initial_values[0]
        initial_volatility = initial_values[1]

    # perform Monte Carlo run and get weights and results
    df_weights, df_results = random_portfolios(data,
                                               num_trials,
                                               riskFreeRate)

    # finding portfolios with the minimum volatility and maximum
    # Sharpe ratio
    index_min_volatility = df_results['Volatility'].idxmin()
    index_max_sharpe = df_results['Sharpe Ratio'].idxmax()
    # storing optimal results to DataFrames
    opt_w = pd.DataFrame([df_weights.iloc[index_min_volatility],
                          df_weights.iloc[index_max_sharpe]],
                         index=['Min Volatility', 'Max Sharpe Ratio'])
    opt_res = pd.DataFrame([df_results.iloc[index_min_volatility],
                            df_results.iloc[index_max_sharpe]],
                           index=['Min Volatility', 'Max Sharpe Ratio'])

    # print out results
    if (verbose):
        opt_vals = ['Min Volatility', 'Max Sharpe Ratio']
        string = ""
        for val in opt_vals:
            string += "-"*70
            string += "\nOptimised portfolio for {}".format(
                val.replace('Min', 'Minimum').replace('Max', 'Maximum'))
            string += "\n\nTime period: {} days".format(freq)
            string += "\nExpected return: {0:0.3f}".format(
                opt_res.loc[val]['Expected Return'])
            string += "\nVolatility: {:0.3f}".format(
                opt_res.loc[val]['Volatility'])
            string += "\nSharpe Ratio: {:0.3f}".format(
                opt_res.loc[val]['Sharpe Ratio'])
            string += "\n\n"+str(opt_w.loc[val].to_frame().transpose().rename(
                    index={val: 'Allocation'}))
            string += "\n"
        string += "-"*70
        print(string)

    # plot results
    if (plot):
        plt.figure()
        # create scatter plot coloured by Sharpe Ratio
        plt.scatter(df_results['Volatility'],
                    df_results['Expected Return'],
                    c=df_results['Sharpe Ratio'],
                    cmap='RdYlBu',
                    s=10,
                    label=None)
        cbar = plt.colorbar()
        # mark in green the minimum volatility
        plt.scatter(opt_res.loc['Min Volatility']['Volatility'],
                    opt_res.loc['Min Volatility']['Expected Return'],
                    marker='^',
                    color='g',
                    s=200,
                    label='min Volatility')
        # mark in red the highest sharpe ratio
        plt.scatter(opt_res.loc['Max Sharpe Ratio']['Volatility'],
                    opt_res.loc['Max Sharpe Ratio']['Expected Return'],
                    marker='^',
                    color='r',
                    s=200,
                    label='max Sharpe Ratio')
        # also set marker for initial portfolio, if weights were given
        if (initial_weights is not None):
            plt.scatter(initial_volatility,
                        initial_return,
                        marker='^',
                        color='k',
                        s=200,
                        label='Initial Portfolio')
        plt.title('Monte Carlo simulation to optimise the portfolio based '
                  + 'on the Efficient Frontier')
        plt.xlabel('Volatility [period='+str(freq)+']')
        plt.ylabel('Expected Return [period='+str(freq)+']')
        cbar.ax.set_ylabel('Sharpe Ratio [period='
                           + str(freq)+']', rotation=90)
        plt.legend()
        plt.show()

    return opt_w, opt_res
