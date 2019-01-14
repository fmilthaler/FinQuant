'''
Provides functions to compute moving averages of data.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeMA(data, fun, spans, plot=True):
    '''
    Computes the moving average (SMA or EWM, depends on the input argument
    "fun") for a number of different time windows.

    Input:
     * data: pandas.DataFrame with stock prices, only one column is expected.
     * fun: function that computes a moving average, e.g. SMA (simple) or
         EWM (exponential).
     * spans: list of integers, time windows to compute the MA on.
     * plot: boolean (default: True), whether to plot the moving averages
         and buy/sell signales based on crossovers of shortest and longest
         moving average.
    '''
    if (not isinstance(data, pd.DataFrame)):
        raise ValueError("data must be of type pandas.DataFrame")
    # compute moving averages
    ma = data.copy(deep=True)
    for span in spans:
        ma[str(span)+"d"] = fun(data, span=span)
    if (plot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plot moving averages
        ma.plot(ax=ax)
        # Create buy/sell signals of shortest and longest span
        minspan = min(spans)
        minlabel = str(minspan)+"d"
        maxspan = max(spans)
        maxlabel = str(maxspan)+"d"
        signals = ma.copy(deep=True)
        signals['diff'] = 0.0
        signals['diff'][minspan:] = np.where(ma[minlabel][minspan:] >
                                             ma[maxlabel][minspan:],
                                             1.0, 0.0)
        # Generate trading orders
        signals['signal'] = signals['diff'].diff()
        # marker for buy signal
        ax.plot(signals.loc[signals['signal'] == 1.0].index, 
                 signals[minlabel][signals['signal'] == 1.0],
                 '^', markersize=8, color='r')
        # marker for sell signal
        ax.plot(signals.loc[signals['signal'] == -1.0].index, 
                 signals[minlabel][signals['signal'] == -1.0],
                 'v', markersize=8, color='b')
        plt.show()
    return ma

def SMA(data, span=100):
    '''
    Computes and returns the simple moving average.
    Note: the moving average is computed on all columns.

    Input:
     * data: pandas.DataFrame with stock prices in columns
     * span: Integer (defaul: 100), number of days/values over which
         the average is computed
    '''
    return data.rolling(window=span, center=False).mean()


def EMA(data, span=100):
    '''
    Computes and returns the exponential moving average.
    Note: the moving average is computed on all columns.

    Input:
     * data: pandas.DataFrame with stock prices in columns
     * span: Integer (defaul: 100), number of days/values over which
         the average is computed
    '''
    return data.ewm(span=span, adjust=False).mean()