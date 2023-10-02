import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from finquant.momentum_indicators import (
    calculate_macd,
    calculate_relative_strength_index,
    calculate_wilder_smoothing_averages,
    gen_macd_color,
    plot_macd,
    plot_relative_strength_index,
)
from finquant.utils import re_download_stock_data

plt.close("all")
plt.switch_backend("Agg")


# Define a sample dataframe for testing
price_data = np.array(
    [100, 102, 105, 103, 108, 110, 107, 109, 112, 115, 120, 118, 121, 124, 125, 126]
).astype(np.float64)
data = pd.DataFrame({"Close": price_data})
macd_data = pd.DataFrame(
    {
        "Date": pd.date_range(start="2022-01-01", periods=16, freq="D"),
        "DIS": price_data,
    }
).set_index("Date", inplace=False)
macd_data.name = "DIS"


def test_calculate_wilder_smoothing_averages():
    # Test with the example values
    result = calculate_wilder_smoothing_averages(10.0, 5.0, 14)
    assert (
        abs(result - 9.642857142857142) <= 1e-15
    )  # The expected result calculated manually

    # Test with zero average gain/loss
    result = calculate_wilder_smoothing_averages(0.0, 5.0, 14)
    assert (
        abs(result - 0.35714285714285715) <= 1e-15
    )  # The expected result calculated manually

    # Test with zero current gain/loss
    result = calculate_wilder_smoothing_averages(10.0, 0.0, 14)
    assert (
        abs(result - 9.285714285714286) <= 1e-15
    )  # The expected result calculated manually

    # Test with window length of 1
    result = calculate_wilder_smoothing_averages(10.0, 5.0, 1)
    assert (
        abs(result - 5.0) <= 1e-15
    )  # Since window length is 1, the result should be the current gain/loss

    # Test with negative values
    result = calculate_wilder_smoothing_averages(-10.0, -5.0, 14)
    assert (
        abs(result - -9.642857142857142) <= 1e-15
    )  # The expected result calculated manually

    # Test with very large numbers
    result = calculate_wilder_smoothing_averages(1e20, 1e20, int(1e20))
    assert (
        abs(result - 1e20) <= 1e-15
    )  # The expected result is the same as input due to the large window length

    # Test with non-float input (should raise an exception)
    with pytest.raises(TypeError):
        calculate_wilder_smoothing_averages("10.0", 5.0, 14)

    # Test with window length of 0 (should raise an exception)
    with pytest.raises(ValueError):
        calculate_wilder_smoothing_averages(10.0, 5.0, 0)

    # Test with negative window length (should raise an exception)
    with pytest.raises(ValueError):
        calculate_wilder_smoothing_averages(10.0, 5.0, -14)


def test_calculate_relative_strength_index():
    rsi = calculate_relative_strength_index(data["Close"])

    # Check if the result is a Pandas Series
    assert isinstance(rsi, pd.Series)

    # Check the length of the result
    assert len(rsi.dropna()) == len(data) - 14

    # Check the first RSI value
    assert np.isclose(rsi.dropna().iloc[0], 82.051282, rtol=1e-4)

    # Check the last RSI value
    assert np.isclose(rsi.iloc[-1], 82.53358925143954, rtol=1e-4)

    # Check that the RSI values are within the range [0, 100]
    assert (rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all()

    # Check for window_length > data length, should raise a ValueError
    with pytest.raises(ValueError):
        calculate_relative_strength_index(data["Close"], window_length=17)

    # Check for oversold >= overbought, should raise a ValueError
    with pytest.raises(ValueError):
        calculate_relative_strength_index(data["Close"], oversold=70, overbought=70)

    # Check for invalid levels, should raise a ValueError
    with pytest.raises(ValueError):
        calculate_relative_strength_index(data["Close"], oversold=150, overbought=80)

    with pytest.raises(ValueError):
        calculate_relative_strength_index(data["Close"], oversold=20, overbought=120)

    # Check for empty input data, should raise a ValueError
    with pytest.raises(ValueError):
        calculate_relative_strength_index(pd.Series([]))

    # Check for non-Pandas Series input, should raise a TypeError
    with pytest.raises(TypeError):
        calculate_relative_strength_index(list(data["Close"]))


def test_plot_relative_strength_index_standalone():
    # Test standalone mode
    xlabel_orig = "Date"
    ylabel_orig = "RSI"
    labels_orig = ["overbought", "oversold", "rsi"]
    title_orig = "RSI Plot"
    plot_relative_strength_index(data["Close"], standalone=True)
    # get data from axis object
    ax = plt.gca()
    # ax.lines[2] is the RSI data
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    # tests
    labels_plot = ax.get_legend_handles_labels()[1]
    title_plot = ax.get_title()
    assert labels_plot == labels_orig
    assert xlabel_plot == xlabel_orig
    assert ylabel_plot == ylabel_orig
    assert title_plot == title_orig


def test_plot_relative_strength_index_not_standalone():
    # Test non-standalone mode
    xlabel_orig = "Date"
    ylabel_orig = "Price"
    plot_relative_strength_index(data["Close"], standalone=False)
    # get data from axis object
    ax = plt.gca()
    line1 = ax.lines[0]
    stock_plot = line1.get_xydata()
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    # tests
    assert (data["Close"].index.values == stock_plot[:, 0]).all()
    assert (data["Close"].values == stock_plot[:, 1]).all()
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot


def test_gen_macd_color_valid_input():
    # Test with valid input
    macd_df = pd.DataFrame({"MACDh": [0.5, -0.2, 0.8, -0.6, 0.2]})
    colors = gen_macd_color(macd_df)

    # Check that the result is a list
    assert isinstance(colors, list)

    # Check the length of the result
    assert len(colors) == len(macd_df)

    # Check color assignments based on MACD values
    assert colors == ["#26A69A", "#FF5252", "#26A69A", "#FF5252", "#26A69A"]


def test_gen_macd_color_green():
    # Test with a DataFrame where MACD values are consistently positive, should return
    # all green colors
    positive_df = pd.DataFrame({"MACDh": [0.5, 0.6, 0.7, 0.8, 0.9]})
    colors = gen_macd_color(positive_df)

    # Check that the result is a list of all green colors
    assert colors == ["#B2DFDB", "#26A69A", "#26A69A", "#26A69A", "#26A69A"]


def test_gen_macd_color_faint_green():
    # Test with a DataFrame where MACD values are consistently positive but decreasing,
    # should return all faint green colors
    faint_green_df = pd.DataFrame({"MACDh": [0.5, 0.4, 0.3, 0.2, 0.1]})
    colors = gen_macd_color(faint_green_df)

    # Check that the result is a list of all faint green colors
    assert colors == ["#26A69A", "#B2DFDB", "#B2DFDB", "#B2DFDB", "#B2DFDB"]


def test_gen_macd_color_red():
    # Test with a DataFrame where MACD values are consistently negative,
    # should return all red colors
    negative_df = pd.DataFrame({"MACDh": [-0.5, -0.6, -0.7, -0.8, -0.9]})
    colors = gen_macd_color(negative_df)

    # Check that the result is a list of all red colors
    assert colors == ["#FFCDD2", "#FF5252", "#FF5252", "#FF5252", "#FF5252"]


def test_gen_macd_color_faint_red():
    # Test with a DataFrame where MACD values are consistently negative but decreasing,
    # should return all faint red colors
    faint_red_df = pd.DataFrame({"MACDh": [-0.5, -0.4, -0.3, -0.2, -0.1]})
    colors = gen_macd_color(faint_red_df)

    # Check that the result is a list of all faint red colors
    assert colors == ["#FF5252", "#FFCDD2", "#FFCDD2", "#FFCDD2", "#FFCDD2"]


def test_gen_macd_color_single_element():
    # Test with a DataFrame containing a single element, should return a list with one color
    single_element_df = pd.DataFrame({"MACDh": [0.5]})
    colors = gen_macd_color(single_element_df)

    # Check that the result is a list with one color
    assert colors == ["#000000"]


def test_gen_macd_color_empty_input():
    # Test with an empty DataFrame, should return an empty list
    empty_df = pd.DataFrame(columns=["MACDh"])
    with pytest.raises(ValueError):
        colors = gen_macd_color(empty_df)


def test_gen_macd_color_missing_column():
    # Test with a DataFrame missing 'MACDh' column, should raise a KeyError
    df_missing_column = pd.DataFrame({"NotMACDh": [0.5, -0.2, 0.8, -0.6, 0.2]})

    with pytest.raises(KeyError):
        gen_macd_color(df_missing_column)


def test_gen_macd_color_no_color_change():
    # Test with a DataFrame where MACD values don't change, should return all black colors
    no_change_df = pd.DataFrame({"MACDh": [0.5, 0.5, 0.5, 0.5, 0.5]})
    colors = gen_macd_color(no_change_df)

    # Check that the result is a list of all black colors
    assert colors == ["#000000", "#000000", "#000000", "#000000", "#000000"]


def test_calculate_macd_valid_input():
    # Test with valid input
    result = calculate_macd(macd_data, num_days_predate_stock_price=0)

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check the length of the result
    assert len(result) == 10
    # not == len(macd_data) here, as we currently re-download data, weekends are not considered

    # Check that the required columns ('MACD', 'MACDh', 'MACDs') are present in the result
    assert all(col in result.columns for col in ["MACD", "MACDh", "MACDs"])


def test_calculate_macd_correct_values():
    # Test for correct values in 'MACD', 'MACDh', and 'MACDs' columns
    longer_ema_window = 10
    shorter_ema_window = 7
    signal_ema_window = 4
    df = re_download_stock_data(
        macd_data, stock_name="DIS", num_days_predate_stock_price=0
    )
    result = calculate_macd(
        macd_data,
        longer_ema_window=longer_ema_window,
        shorter_ema_window=shorter_ema_window,
        signal_ema_window=signal_ema_window,
        num_days_predate_stock_price=0,
    )

    # Calculate expected values manually (using the provided df)
    ema_short = (
        df["Close"]
        .ewm(span=shorter_ema_window, adjust=False, min_periods=shorter_ema_window)
        .mean()
    )
    ema_long = (
        df["Close"]
        .ewm(span=longer_ema_window, adjust=False, min_periods=longer_ema_window)
        .mean()
    )
    macd = ema_short - ema_long
    macd.name = "MACD"
    signal = macd.ewm(
        span=signal_ema_window, adjust=False, min_periods=signal_ema_window
    ).mean()
    macd_h = macd - signal

    # Check that the calculated values match the values in the DataFrame
    assert all(result["MACD"].dropna() == macd.dropna())
    assert all(result["MACDh"].dropna() == macd_h.dropna())
    assert all(result["MACDs"].dropna() == signal.dropna())


def test_calculate_macd_custom_windows():
    # Test with custom EMA window values
    result = calculate_macd(
        macd_data, longer_ema_window=30, shorter_ema_window=15, signal_ema_window=10
    )

    # Check that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the required columns ('MACD', 'MACDh', 'MACDs') are present in the result
    assert all(col in result.columns for col in ["MACD", "MACDh", "MACDs"])


def test_calculate_macd_invalid_windows():
    # Test with invalid window values, should raise ValueError
    with pytest.raises(ValueError):
        calculate_macd(
            macd_data, longer_ema_window=10, shorter_ema_window=20, signal_ema_window=15
        )
    with pytest.raises(ValueError):
        plot_macd(
            macd_data, longer_ema_window=10, shorter_ema_window=5, signal_ema_window=30
        )


def test_plot_macd():
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
