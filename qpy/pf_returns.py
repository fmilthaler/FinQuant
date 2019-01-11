'''
Providing functions to compute daily returns of stock data
'''
import numpy as np
import pandas as pd


def simpleReturns(data):
    '''
    Returns DataFrame with returns
    price_{t} / price_{t=0}

    Input:
     * data: DataFrame with daily stock prices

    Output: DataFrame of daily percentage change of returns
    of given stock prices
    '''
    return data.apply(lambda x: x / x[0])


def dailyReturns(data):
    '''
    Returns DataFrame with daily returns

    Input:
     * data: DataFrame with daily stock prices

    Output: DataFrame of daily percentage change of returns
    of given stock prices
    '''
    return data.pct_change().dropna(how="all")


def dailyLogReturns(data):
    '''
    Returns DataFrame with daily log returns

    Input:
     * data: DataFrame with daily stock prices

    Output: DataFrame of log(1 + daily percentage change of returns)
    '''
    return np.log(1 + dailyReturns(data)).dropna(how="all")


def historicalMeanReturn(data, freq=252):
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
    daily_returns = dailyReturns(data)
    return daily_returns.mean() * freq
