'''
Provides functions to compute moving averages of data.
'''
import numpy as np
import pandas as pd


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


def EWM(data, span=100):
    '''
    Computes and returns the exponential moving average.
    Note: the moving average is computed on all columns.

    Input:
     * data: pandas.DataFrame with stock prices in columns
     * span: Integer (defaul: 100), number of days/values over which
         the average is computed
    '''
    return data.ewm(span=span, adjust=False).mean()
