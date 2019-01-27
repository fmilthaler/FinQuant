'''
Provides functions to compute daily returns of stock data
'''


import numpy as np
import pandas as pd


def cumulative_returns(data, dividend=0):
    '''
    Returns DataFrame with cumulative returns
    R = (price_{t_i} - price_{t_0} + dividend) / price_{t_0}

    Input:
     * data: DataFrame with daily stock prices
     * dividend: Float (default: 0), paid dividend

    Output: DataFrame of cumulative returns of given stock prices
    '''
    return data.apply(lambda x: (x - x[0] + dividend) / x[0])


def daily_returns(data):
    '''
    Returns DataFrame with daily returns (percentage change)
    R = (price_{t_i} - price_{t_{i-1}}) / price_{t_{i-1}}

    Input:
     * data: DataFrame with daily stock prices

    Output: DataFrame of daily percentage change of returns
    of given stock prices
    '''
    return data.pct_change().dropna(how="all")


def daily_log_returns(data):
    '''
    Returns DataFrame with daily log returns
    R_{log} = log(1 + (price_{t_i} - price_{t_{i-1}}) / price_{t_{i-1}})

    Input:
     * data: DataFrame with daily stock prices

    Output: DataFrame of log(1 + daily percentage change of returns)
    '''
    return np.log(1 + daily_returns(data)).dropna(how="all")


def historical_mean_return(data, freq=252):
    '''
    Returns the mean return based on historical stock price data.

    Input:
     * data: DataFrame with daily stock prices
     * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year

    Output: DataFrame of mean daily * freq
    '''
    if (not isinstance(data, pd.DataFrame)):
        raise ValueError("data must be a pandas.DataFrame")
    return daily_returns(data).mean() * freq
