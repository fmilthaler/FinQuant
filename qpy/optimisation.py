import numpy as np
import pandas as pd
from qpy.pf_returns import dailyReturns
from qpy.pf_quants import weightedMean, weightedStd, sharpeRatio


def optimisePfMC(data,
                 total_investment,
                 num_trials=10000,
                 riskFreeRate=0.005,
                 freq=252,
                 initial_weights=None,
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
     * plot: Boolean (default: True), if True, a plot showing the results
         is produced

    Output:
     * pf_opt: DataFrame with optimised investment strategies for maximum
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
    res_columns.extend(['Return', 'Volatility', 'Sharpe Ratio'])
    results = np.zeros((len(res_columns), num_trials))
    # compute returns, means and covariance matrix
    returns = dailyReturns(data)
    pf_return_means = returns.mean()
    cov_matrix = returns.cov()

    # computed expected return and volatility of initial portfolio
    if (initial_weights is not None):
        initial_pf_return = weightedMean(pf_return_means,
                                         initial_weights) * freq
        initial_pf_volatility = weightedStd(cov_matrix,
                                            initial_weights
                                            ) * np.sqrt(freq)

    # Monte Carlo simulation
    for i in range(num_trials):
        # select random weights for portfolio
        weights = np.array(np.random.random(num_stocks))
        # rebalance weights
        weights = weights/np.sum(weights)
        # compute portfolio return and volatility
        pf_return = weightedMean(pf_return_means, weights) * freq
        pf_volatility = weightedStd(cov_matrix, weights) * np.sqrt(freq)

        # add weights times total_investments to results array
        results[0:num_stocks, i] = weights*total_investment
        # store results in results array
        results[num_stocks, i] = pf_return
        results[num_stocks+1, i] = pf_volatility
        results[num_stocks+2, i] = sharpeRatio(pf_return,
                                               pf_volatility,
                                               riskFreeRate)

    # transpose and convert to pandas.DataFrame:
    df_results = pd.DataFrame(results.T, columns=res_columns)
    # adding info of max Sharpe ratio and of min volatility
    # to resulting df (with meaningful indices):
    pf_opt = pd.DataFrame([df_results.iloc[
        df_results['Sharpe Ratio'].idxmax()],
        df_results.iloc[df_results['Volatility'].idxmin()]],
        index=['Max Sharpe Ratio', 'Min Volatility'])

    # plot results
    if (plot):
        import matplotlib.pylab as plt
        plt.figure()
        # create scatter plot coloured by Sharpe Ratio
        plt.scatter(df_results['Volatility'],
                    df_results['Return'],
                    c=df_results['Sharpe Ratio'],
                    cmap='RdYlBu',
                    s=10,
                    label=None)
        cbar = plt.colorbar()
        # mark in red the highest sharpe ratio
        plt.scatter(pf_opt.loc['Max Sharpe Ratio']['Volatility'],
                    pf_opt.loc['Max Sharpe Ratio']['Return'],
                    marker='^',
                    color='r',
                    s=200,
                    label='max Sharpe Ratio')
        # mark in green the minimum volatility
        plt.scatter(pf_opt.loc['Min Volatility']['Volatility'],
                    pf_opt.loc['Min Volatility']['Return'],
                    marker='^',
                    color='g',
                    s=200,
                    label='min Volatility')
        # also set marker for initial portfolio, if weights were given
        if (initial_weights is not None):
            plt.scatter(initial_pf_volatility,
                        initial_pf_return,
                        marker='^',
                        color='k',
                        s=200,
                        label='Initial Portfolio')
        plt.title('Monte Carlo simulation to optimise the portfolio based '
                  + 'on the Efficient Frontier')
        plt.xlabel('Volatility [period='+str(freq)+']')
        plt.ylabel('Returns [period='+str(freq)+']')
        cbar.ax.set_ylabel('Sharpe Ratio [period='
                           + str(freq)+']', rotation=90)
        plt.legend()
        plt.show()

    return pf_opt
