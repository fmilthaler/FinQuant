import numpy as np
import pandas as pd


def weightedMean(means, weights):
    return np.sum(means * weights)


def weightedStd(cov_matrix, weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def SharpeRatio(exproi, riskfreerate, volatility):
    return (exproi - riskfreerate)/float(volatility)


def optimisePortfolio(roi_data, total_investment, num_trials=10000,
                      riskfreerate=0.005, period=252,
                      plot=True):
        '''
        Optimisation of the portfolio by performing a Monte Carlo simulation.

        Input:
         * roi_data: A DataFrame which contains the return of investment (ROI)
             data of relevant stocks
         * total_investment: Float, money to be invested.
         * num_trials: Integer (default: 10000), number of portfolios to be computed, each with a random distribution of weights/investments in each stock
         * riskfreerate: Float (default: 0.005), the risk free rate as required for the Sharpe Ratio
         * period: Integer (default: 252), number of trading days, default value corresponds to trading days in a year
         * plot: Boolean (default: True), if True, a plot showing the results is produced
        Output:
         * pf_opt: DataFrame with optimised investment strategies for maximum Sharpe Ratio and minimum volatility.
        '''
        # set number of stocks in the portfolio
        num_stocks = len(roi_data.columns)
        # set up array to hold results
        res_columns = list(roi_data.columns)
        res_columns.extend(['ROI', 'Volatility', 'Sharpe Ratio'])
        results = np.zeros((len(res_columns), num_trials))
        # compute means and covariance matrix
        pf_means = roi_data.mean().values
        cov_matrix = roi_data.cov()
        # monte carlo simulation
        for i in range(num_trials):
            # select random weights for portfolio
            weights = np.array(np.random.random(num_stocks))
            # rebalance weights
            weights = weights/np.sum(weights)
            # compute portfolio roi and volatility
            pf_roi = weightedMean(pf_means, weights) * period
            pf_volatility = weightedStd(cov_matrix, weights) * np.sqrt(period)

            # add weights times total_investments to results array
            results[0:num_stocks, i] = weights*total_investment
            # store results in results array
            results[num_stocks, i] = pf_roi
            results[num_stocks+1, i] = pf_volatility
            results[num_stocks+2, i] = SharpeRatio(pf_roi,
                                                   riskfreerate,
                                                   pf_volatility)

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
                        df_results['ROI'],
                        c=df_results['Sharpe Ratio'],
                        cmap='RdYlBu',
                        label=None
                       )
            plt.colorbar()
            # mark in red the highest sharpe ratio
            plt.scatter(pf_opt.loc['Max Sharpe Ratio']['Volatility'],
                        pf_opt.loc['Max Sharpe Ratio']['ROI'],
                        marker='^',
                        color='r',
                        s=250,
                        label='max Sharpe Ratio'
                       )
            # mark in green the minimum volatility
            plt.scatter(pf_opt.loc['Min Volatility']['Volatility'],
                        pf_opt.loc['Min Volatility']['ROI'],
                        marker='^',
                        color='g',
                        s=250,
                        label='min Volatility',
                       )
            plt.title('Monte Carlo simulation to optimise the portfolio')
            plt.xlabel('Volatility')
            plt.ylabel('ROI [period='+str(period)+']')
            plt.legend()
            plt.show()

        return pf_opt
