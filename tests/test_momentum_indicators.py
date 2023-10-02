import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from finquant.momentum_indicators import plot_macd
from finquant.momentum_indicators import relative_strength_index as rsi

plt.close("all")
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
    line1 = ax.lines[0]
    stock_plot = line1.get_xydata()
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    # tests
    assert (df["Stock"].index.values == stock_plot[:, 0]).all()
    assert (df["Stock"].values == stock_plot[:, 1]).all()
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot


def test_rsi_standalone():
    x = np.sin(np.linspace(1, 10, 100))
    xlabel_orig = "Date"
    ylabel_orig = "RSI"
    labels_orig = ["overbought", "oversold", "rsi"]
    title_orig = "RSI Plot"
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
    print(xlabel_plot, ylabel_plot)
    # tests
    assert (df["rsi"].index.values == rsi_plot[:, 0]).all()
    # for comparing values, we need to remove nan
    a, b = df["rsi"].values, rsi_plot[:, 1]
    a, b = map(lambda x: x[~np.isnan(x)], (a, b))
    assert (a == b).all()
    labels_plot = ax.get_legend_handles_labels()[1]
    title_plot = ax.get_title()
    assert labels_plot == labels_orig
    assert xlabel_plot == xlabel_orig
    assert ylabel_plot == ylabel_orig
    assert title_plot == title_orig


def test_mpl_macd():
    axes0_ylabel_orig = "Price"
    axes4_ylabel_orig = "Volume  $10^{6}$"
    # Create sample data for testing
    x = np.sin(np.linspace(1, 10, 100))
    df = pd.DataFrame(
        {"Close": x}, index=pd.date_range("2015-01-01", periods=100, freq="D")
    )
    df.name = "DIS"

    # Call mpl_macd function
    fig, axes = plot_macd(df)

    axes0_ylabel_plot = axes[0].get_ylabel()
    axes4_ylabel_plot = axes[4].get_ylabel()

    # Check if the function returned valid figures and axes objects
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 6  # Assuming there are six subplots in the returned figure
    assert axes0_ylabel_orig == axes0_ylabel_plot
    assert axes4_ylabel_orig == axes4_ylabel_plot


def test_mpl_macd_invalid_window_parameters():
    # Create sample data with invalid window parameters
    x = np.sin(np.linspace(1, 10, 100))
    df = pd.DataFrame(
        {"Close": x}, index=pd.date_range("2015-01-01", periods=100, freq="D")
    )
    df.name = "DIS"

    # Call mpl_macd function with invalid window parameters and check for ValueError
    with pytest.raises(ValueError):
        plot_macd(df, longer_ema_window=10, shorter_ema_window=20, signal_ema_window=30)
    with pytest.raises(ValueError):
        plot_macd(df, longer_ema_window=10, shorter_ema_window=5, signal_ema_window=30)
