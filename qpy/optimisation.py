'''
Provides optimisation functions
'''
import numpy as np
import pandas as pd
from qpy.returns import dailyReturns
from qpy.quants import weightedMean, weightedStd, sharpeRatio
from qpy.quants import annualised_portfolio_quantities


def random_portfolios(data, num_trials, riskFreeRate, freq=252):
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
    # set number of stocks in the portfolio
    num_stocks = len(data.columns)
    # set up array to hold results
    res_columns = list(data.columns)
    res_columns.extend(['Expected Return', 'Volatility', 'Sharpe Ratio'])
    results = np.zeros((len(res_columns), num_trials))
    # compute returns, means and covariance matrix
    returns = dailyReturns(data)
    return_means = returns.mean()
    cov_matrix = returns.cov()

    # computed expected return and volatility of initial portfolio
    if (initial_weights is not None):
        initial_return = weightedMean(return_means,
                                      initial_weights) * freq
        initial_volatility = weightedStd(cov_matrix,
                                         initial_weights
                                         ) * np.sqrt(freq)

    # Monte Carlo simulation
    for i in range(num_trials):
        # select random weights for portfolio
        weights = np.array(np.random.random(num_stocks))
        # rebalance weights
        weights = weights/np.sum(weights)
        # compute portfolio return and volatility
        expectedReturn = weightedMean(return_means, weights) * freq
        volatility = weightedStd(cov_matrix, weights) * np.sqrt(freq)

        # weights times total_investments = money to be invested
        # in each stock, but here, weights should remain relative
        # to the sum of weights
        results[0:num_stocks, i] = weights#*total_investment
        # store results in results array
        results[num_stocks, i] = expectedReturn
        results[num_stocks+1, i] = volatility
        results[num_stocks+2, i] = sharpeRatio(expectedReturn,
                                               volatility,
                                               riskFreeRate)

    # transpose and convert to pandas.DataFrame:
    df_results = pd.DataFrame(results.T, columns=res_columns)
    # adding info of max Sharpe ratio and of min volatility
    # to resulting df (with meaningful indices):
    opt = pd.DataFrame([df_results.iloc[
        df_results['Sharpe Ratio'].idxmax()],
        df_results.iloc[df_results['Volatility'].idxmin()]],
        index=['Max Sharpe Ratio', 'Min Volatility'])

    # print out results
    if (verbose):
        opt_vals = ['Min Volatility', 'Max Sharpe Ratio']
        string = ""
        for val in opt_vals:
            string += "-"*70
            string += "\nOptimised portfolio for {}".format(val.replace('Min', 'Minimum').replace('Max', 'Maximum'))
            string += "\n\nTime period: {} days".format(freq)
            string += "\nExpected return: {0:0.3f}".format(opt.loc[val]['Expected Return'])
            string += "\nVolatility: {:0.3f}".format(opt.loc[val]['Volatility'])
            string += "\n\n"+str(opt.loc[val].iloc[0:num_stocks].to_frame().transpose().rename(index={val: 'Allocation'}))
            string += "\n"
        string += "-"*70
        print(string)

    # plot results
    if (plot):
        import matplotlib.pylab as plt
        plt.figure()
        # create scatter plot coloured by Sharpe Ratio
        plt.scatter(df_results['Volatility'],
                    df_results['Expected Return'],
                    c=df_results['Sharpe Ratio'],
                    cmap='RdYlBu',
                    s=10,
                    label=None)
        cbar = plt.colorbar()
        # mark in red the highest sharpe ratio
        plt.scatter(opt.loc['Max Sharpe Ratio']['Volatility'],
                    opt.loc['Max Sharpe Ratio']['Expected Return'],
                    marker='^',
                    color='r',
                    s=200,
                    label='max Sharpe Ratio')
        # mark in green the minimum volatility
        plt.scatter(opt.loc['Min Volatility']['Volatility'],
                    opt.loc['Min Volatility']['Expected Return'],
                    marker='^',
                    color='g',
                    s=200,
                    label='min Volatility')
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

    return opt
