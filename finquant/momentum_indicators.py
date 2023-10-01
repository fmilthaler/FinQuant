""" This module provides function(s) to compute momentum indicators
used in technical analysis such as RSI, MACD etc. """
import datetime
from typing import List

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from finquant.data_types import INT, SERIES_OR_DATAFRAME
from finquant.type_utilities import type_validation
from finquant.utils import all_list_ele_in_other
from finquant.portfolio import _yfinance_request

def calculate_wilder_smoothing_averages(avg_gain_loss, gain_loss, window_length):
    return (avg_gain_loss * (window_length - 1) + gain_loss) / window_length


def relative_strength_index(
    data: SERIES_OR_DATAFRAME,
    window_length: INT = 14,
    oversold: INT = 30,
    overbought: INT = 70,
    standalone: bool = False,
) -> None:
    """Computes and visualizes a RSI graph,
        plotted along with the prices in another sub-graph
        for comparison.

    Ref: https://www.investopedia.com/terms/r/rsi.asp

    :param data: A series/dataframe of daily stock prices
    :type data: :py:data:`~.finquant.data_types.SERIES_OR_DATAFRAME`
    :param window_length: Window length to compute RSI, default being 14 days
    :type window_length: :py:data:`~.finquant.data_types.INT`
    :param oversold: Standard level for oversold RSI, default being 30
    :type oversold: :py:data:`~.finquant.data_types.INT`
    :param overbought: Standard level for overbought RSI, default being 70
    :type overbought: :py:data:`~.finquant.data_types.INT`
    :param standalone: Plot only the RSI graph
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError(
            "data is expected to be of type pandas.Series or pandas.DataFrame"
        )
    if isinstance(data, pd.DataFrame) and not len(data.columns.values) == 1:
        raise ValueError("data is expected to have only one column.")
    # checking integer fields
    for field in (window_length, oversold, overbought):
        if not isinstance(field, int):
            raise ValueError(f"{field} must be an integer.")
    # validating levels
    if oversold >= overbought:
        raise ValueError("oversold level should be < overbought level")
    if not 0 < oversold < 100 or not 0 < overbought < 100:
        raise ValueError("levels should be > 0 and < 100")
    # converting data to pd.DataFrame if it is a pd.Series (for subsequent function calls):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # calculate price differences
    data["diff"] = data.diff(periods=1)
    # calculate gains and losses
    data["gain"] = data["diff"].clip(lower=0)
    data["loss"] = data["diff"].clip(upper=0).abs()
    # calculate rolling window mean gains and losses
    data["avg_gain"] = (
        data["gain"].rolling(window=window_length, min_periods=window_length).mean()
    )
    data["avg_loss"] = (
        data["loss"].rolling(window=window_length, min_periods=window_length).mean()
    )
    # ignore SettingWithCopyWarning for the below operation
    with pd.option_context("mode.chained_assignment", None):
        for gain_or_loss in ["gain", "loss"]:
            for idx, _ in enumerate(
                data[f"avg_{gain_or_loss}"].iloc[window_length + 1 :]
            ):
                data[f"avg_{gain_or_loss}"].iloc[
                    idx + window_length + 1
                ] = calculate_wilder_smoothing_averages(
                    data[f"avg_{gain_or_loss}"].iloc[idx + window_length],
                    data[gain_or_loss].iloc[idx + window_length + 1],
                    window_length,
                )
    # calculate RS values
    data["rs"] = data["avg_gain"] / data["avg_loss"]
    # calculate RSI
    data["rsi"] = 100 - (100 / (1.0 + data["rs"]))
    # Plot it
    stock_name = data.keys()[0]
    if standalone:
        # Single plot
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.axhline(y=overbought, color="r", linestyle="dashed", label="overbought")
        axis.axhline(y=oversold, color="g", linestyle="dashed", label="oversold")
        axis.set_ylim(0, 100)
        data["rsi"].plot(ylabel="RSI", xlabel="Date", ax=axis, grid=True)
        plt.title("RSI Plot")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        # RSI against price in 2 plots
        fig, axis = plt.subplots(2, 1, sharex=True, sharey=False)
        axis[0].axhline(y=overbought, color="r", linestyle="dashed", label="overbought")
        axis[0].axhline(y=oversold, color="g", linestyle="dashed", label="oversold")
        axis[0].set_title("RSI + Price Plot")
        axis[0].set_ylim(0, 100)
        # plot 2 graphs in 2 colors
        colors = plt.rcParams["axes.prop_cycle"]()
        data["rsi"].plot(
            ylabel="RSI", ax=axis[0], grid=True, color=next(colors)["color"], legend=True
        ).legend(loc="center left", bbox_to_anchor=(1, 0.5))
        data[stock_name].plot(
            xlabel="Date",
            ylabel="Price",
            ax=axis[1],
            grid=True,
            color=next(colors)["color"],
            legend=True,
        ).legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return data["rsi"]


# Generating colors for MACD histogram
def gen_macd_color(df: pd.DataFrame) -> List[str]:
    """
    Generate a list of color codes based on MACD histogram values in a DataFrame.

    This function takes a DataFrame containing MACD histogram values ('MACDh') and
    assigns colors to each data point based on the direction of change in MACD values.

    :param df: A series/dataframe of MACD histogram values

    :return: A list of color codes corresponding to each data point in the DataFrame.

    Note:
    - This function assumes that the DataFrame contains a column named 'MACDh'.
    - The color assignments are based on the comparison of each data point with its
      previous data point in the 'MACDh' column.

    Example usage:
    ```
    import pandas as pd
    from typing import List

    # Create a DataFrame with MACD histogram values
    df = pd.DataFrame({'MACDh': [0.5, -0.2, 0.8, -0.6, 0.2]})

    # Generate MACD color codes
    colors = gen_macd_color(df)
    print(colors)  # Output: ['#26A69A', '#FFCDD2', '#26A69A', '#FFCDD2', '#26A69A']
    ```
    """
    type_validation(df=df)
    macd_color = []
    macd_color.clear()
    for idx in range(0, len(df["MACDh"])):
        if df["MACDh"].iloc[idx] >= 0 and df["MACDh"].iloc[idx - 1] < df["MACDh"].iloc[idx]:
            macd_color.append("#26A69A")  # green
        elif df["MACDh"].iloc[idx] >= 0 and df["MACDh"].iloc[idx - 1] > df["MACDh"].iloc[idx]:
            macd_color.append("#B2DFDB")  # faint green
        elif df["MACDh"].iloc[idx] < 0 and df["MACDh"].iloc[idx - 1] > df["MACDh"].iloc[idx]:
            macd_color.append("#FF5252")  # red
        elif df["MACDh"].iloc[idx] < 0 and df["MACDh"].iloc[idx - 1] < df["MACDh"].iloc[idx]:
            macd_color.append("#FFCDD2")  # faint red
        else:
            macd_color.append("#000000")
    return macd_color


def mpl_macd(
    data: SERIES_OR_DATAFRAME,
    longer_ema_window: INT = 26,
    shorter_ema_window: INT = 12,
    signal_ema_window: INT = 9,
    stock_name: str = None,
):
    """
    Generate a Matplotlib candlestick chart with MACD (Moving Average Convergence Divergence) indicators.

    Ref: https://github.com/matplotlib/mplfinance/blob/master/examples/indicators/macd_histogram_gradient.ipynb

    This function creates a candlestick chart using the given stock price data and overlays
    MACD, MACD Signal Line, and MACD Histogram indicators. The MACD is calculated by taking
    the difference between two Exponential Moving Averages (EMAs) of the closing price.


    :param data: Time series data containing stock price information. If a
      DataFrame is provided, it should have columns 'Open', 'Close', 'High', 'Low', and 'Volume'.
      Else, stock price data for given time frame is downloaded again.
    :type data: :py:data:`~.finquant.data_types.SERIES_OR_DATAFRAME`
    :param longer_ema_window: Optional, window size for the longer-term EMA (default is 26).
    :type longer_ema_window: :py:data:`~.finquant.data_types.INT`
    :param shorter_ema_window: Optional, window size for the shorter-term EMA (default is 12).
    :type shorter_ema_window: :py:data:`~.finquant.data_types.INT`
    :param signal_ema_window: Optional, window size for the signal line EMA (default is 9).
    :type signal_ema_window: :py:data:`~.finquant.data_types.INT`
    :param stock_name: Optional, name of the stock for labeling purposes (default is None).

    Note:
    - If the input data is a DataFrame, it should contain columns 'Open', 'Close', 'High', 'Low', and 'Volume'.
    - If the input data is a Series, it should have a valid name.
    - The longer EMA window should be greater than or equal to the shorter EMA window and signal EMA window.

    Example usage:
    ```
    import pandas as pd
    from mplfinance.original_flavor import plot as mpf

    # Create a DataFrame or Series with stock price data
    data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
    mpl_macd(data, longer_ema_window=26, shorter_ema_window=12, signal_ema_window=9, stock_name='AAPL')
    ```
    """
    # Type validations:
    type_validation(
        data=data,
        longer_ema_window=longer_ema_window,
        shorter_ema_window=shorter_ema_window,
        signal_ema_window=signal_ema_window,
        name=stock_name,
    )

    # validating windows
    if longer_ema_window < shorter_ema_window:
        raise ValueError("longer ema window should be > shorter ema window")
    if longer_ema_window < signal_ema_window:
        raise ValueError("longer ema window should be > signal ema window")

    # Taking care of potential column header clash, removing "WIKI/" (which comes from legacy quandl)
    if stock_name is None:
        stock_name = data.name
    if "WIKI/" in stock_name:
        stock_name = stock_name.replace("WIKI/", "")
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Remove prefix substring from column headers
    data.columns = data.columns.str.replace("WIKI/", "")

    # Check if required columns are present, if data is a pd.DataFrame, else re-download stock price data:
    re_download_stock_data = True
    if isinstance(data, pd.DataFrame) and all_list_ele_in_other(
        ["Open", "Close", "High", "Low", "Volume"], data.columns
    ):
        re_download_stock_data = False
    if re_download_stock_data:
        # download additional price data 'Open' for given stock and timeframe:
        start_date = data.index.min() - datetime.timedelta(days=31)
        end_date = data.index.max() + datetime.timedelta(days=1)
        df = _yfinance_request([stock_name], start_date=start_date, end_date=end_date)
        # dropping second level of column header that yfinance returns
        df.columns = df.columns.droplevel(1)
    else:
        df = data

    # Get the shorter_ema_window-day EMA of the closing price
    macd_k = (
        df["Close"]
        .ewm(span=shorter_ema_window, adjust=False, min_periods=shorter_ema_window)
        .mean()
    )
    # Get the longer_ema_window-day EMA of the closing price
    macd_d = (
        df["Close"]
        .ewm(span=longer_ema_window, adjust=False, min_periods=longer_ema_window)
        .mean()
    )

    # Subtract the longer_ema_window-day EMA from the shorter_ema_window-Day EMA to get the MACD
    macd = macd_k - macd_d
    # Get the signal_ema_window-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(
        span=signal_ema_window, adjust=False, min_periods=signal_ema_window
    ).mean()
    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s

    # Add all of our new values for the MACD to the dataframe
    df["MACD"] = df.index.map(macd)
    df["MACDh"] = df.index.map(macd_h)
    df["MACDs"] = df.index.map(macd_s)

    # plot macd
    macd_color = gen_macd_color(df)
    apds = [
        mpf.make_addplot(macd, color="#2962FF", panel=1),
        mpf.make_addplot(macd_s, color="#FF6D00", panel=1),
        mpf.make_addplot(
            macd_h,
            type="bar",
            width=0.7,
            panel=1,
            color=macd_color,
            alpha=1,
            secondary_y=True,
        ),
    ]
    fig, axes = mpf.plot(
        df,
        volume=True,
        type="candle",
        style="yahoo",
        addplot=apds,
        volume_panel=2,
        figsize=(20, 10),
        returnfig=True,
    )
    axes[2].legend(["MACD"], loc="upper left")
    axes[3].legend(["Signal"], loc="lower left")

    return fig, axes
