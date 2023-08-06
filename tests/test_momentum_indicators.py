import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finquant.momentum_indicators import (
    relative_strength_index as rsi,
    macd,
)

plt.switch_backend("Agg")

def test_rsi():
    x = np.sin(np.linspace(1, 10, 100))
    xlabel_orig = "Date"
    ylabel_orig = "Price"
    df = pd.DataFrame({"Stock": x}, index=np.linspace(1, 10, 100))
    df.index.name = "Date"
    rsi(df)
    # get data from axis object
    ax = plt.gca()
    # ax.lines[0] is the data we passed to plot_bollinger_band    
    line1 = ax.lines[0]
    stock_plot = line1.get_xydata()
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    # tests
    assert (df['Stock'].index.values == stock_plot[:, 0]).all()
    assert (df["Stock"].values == stock_plot[:, 1]).all()
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot
    
def test_rsi_standalone():
    x = np.sin(np.linspace(1, 10, 100))
    xlabel_orig = "Date"
    ylabel_orig = "RSI"
    labels_orig = ['rsi']
    title_orig = 'RSI Plot'
    df = pd.DataFrame({"Stock": x}, index=np.linspace(1, 10, 100))
    df.index.name = "Date"
    rsi(df, standalone=True)
    # get data from axis object
    ax = plt.gca()
    # ax.lines[2] is the RSI data
    line1 = ax.lines[2]
    rsi_plot = line1.get_xydata()
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    print (xlabel_plot, ylabel_plot)    
    # tests
    assert (df['rsi'].index.values == rsi_plot[:, 0]).all()
    # for comparing values, we need to remove nan
    a, b = df['rsi'].values, rsi_plot[:, 1]
    a, b = map(lambda x: x[~np.isnan(x)], (a, b))
    assert (a == b).all()
    labels_plot = ax.get_legend_handles_labels()[1]
    title_plot = ax.get_title()
    assert labels_plot == labels_orig
    assert xlabel_plot == xlabel_orig
    assert ylabel_plot == ylabel_orig    
    assert title_plot == title_orig

def test_macd():
    x = np.sin(np.linspace(1, 10, 100))
    xlabel_orig = "Date"
    ylabel_orig = "Price"
    df = pd.DataFrame({"Stock": x}, index=np.linspace(1, 10, 100))
    df.index.name = "Date"
    macd(df)
    # get data from axis object
    ax = plt.gca()
    # ax.lines[0] is the data we passed to plot_bollinger_band    
    line1 = ax.lines[0]
    stock_plot = line1.get_xydata()
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    # tests
    assert (df['Stock'].index.values == stock_plot[:, 0]).all()
    assert (df["Stock"].values == stock_plot[:, 1]).all()
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot
    
def test_macd_standalone():
    labels_orig = ['MACD', 'diff', 'SIGNAL']    
    x = np.sin(np.linspace(1, 10, 100))
    xlabel_orig = "Date"
    ylabel_orig = "MACD"
    df = pd.DataFrame({"Stock": x}, index=np.linspace(1, 10, 100))
    df.index.name = "Date"
    macd(df, standalone=True)
    # get data from axis object
    ax = plt.gca()
    labels_plot = ax.get_legend_handles_labels()[1]
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    assert labels_plot == labels_orig
    assert xlabel_plot == xlabel_orig
    assert ylabel_plot == ylabel_orig    
    # ax.lines[0] is macd data
    # ax.lines[1] is diff data
    # ax.lines[2] is macd_s data
    # tests
    for i, key in ((0, 'macd'), (1, 'diff'), (2, 'macd_s')):
        line = ax.lines[i]
        data_plot = line.get_xydata()
        # tests
        assert (df[key].index.values == data_plot[:, 0]).all()
        # for comparing values, we need to remove nan
        a, b = df[key].values, data_plot[:, 1]
        a, b = map(lambda x: x[~np.isnan(x)], (a, b))
        assert (a == b).all()
