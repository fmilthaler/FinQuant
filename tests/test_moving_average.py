import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finquant.moving_average import compute_ma, sma, ema, sma_std, ema_std
from finquant.moving_average import plot_bollinger_band


def test_sma():
    orig = np.array(
        [
            [np.nan, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            [np.nan, 0.5, 2.5, 6.5, 12.5, 20.5, 30.5, 42.5, 56.5, 72.5],
        ]
    )
    dforig = pd.DataFrame({"0": orig[0], "1": orig[1]}).dropna()
    l1 = range(10)
    l2 = [i ** 2 for i in range(10)]
    df = pd.DataFrame({"0": l1, "1": l2})
    res = sma(df, span=2).dropna()
    assert all((dforig == res).all())


def test_ema():
    orig = np.array(
        [
            [
                np.nan,
                0.6666666666666666,
                1.5555555555555556,
                2.5185185185185186,
                3.506172839506173,
                4.502057613168724,
                5.500685871056241,
                6.500228623685413,
                7.5000762078951375,
                8.500025402631714,
            ],
            [
                np.nan,
                0.6666666666666666,
                2.888888888888889,
                6.962962962962963,
                12.987654320987653,
                20.99588477366255,
                30.998628257887518,
                42.99954275262917,
                56.99984758420972,
                72.99994919473657,
            ],
        ]
    )
    dforig = pd.DataFrame({"0": orig[0], "1": orig[1]}).dropna()
    l1 = range(10)
    l2 = [i ** 2 for i in range(10)]
    df = pd.DataFrame({"0": l1, "1": l2})
    res = ema(df, span=2).dropna()
    assert all((abs(dforig - res) <= 1e-15).all())


def test_sma_std():
    orig = np.array(
        [
            [
                np.nan,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
            ],
            [
                np.nan,
                0.7071067811865476,
                2.1213203435596424,
                3.5355339059327378,
                4.949747468305833,
                6.363961030678928,
                7.7781745930520225,
                9.192388155425117,
                10.606601717798213,
                12.020815280171307,
            ],
        ]
    )
    dforig = pd.DataFrame({"0": orig[0], "1": orig[1]}).dropna()
    l1 = range(10)
    l2 = [i ** 2 for i in range(10)]
    df = pd.DataFrame({"0": l1, "1": l2})
    res = sma_std(df, span=2).dropna()
    assert all((abs(dforig - res) <= 1e-15).all())


def test_ema_std():
    orig = np.array(
        [
            [
                np.nan,
                0.7071067811865476,
                0.9746794344808964,
                1.1143420667632726,
                1.1785687889316867,
                1.20612962779329,
                1.217443715603457,
                1.2219416913579804,
                1.2236866244000921,
                1.2243507269461653,
            ],
            [
                np.nan,
                0.7071067811865476,
                2.2693611435820435,
                4.280032864205755,
                6.511621880314852,
                8.846731940915395,
                11.231335395956103,
                13.640730921938678,
                16.063365414263,
                18.493615652686387,
            ],
        ]
    )
    dforig = pd.DataFrame({"0": orig[0], "1": orig[1]}).dropna()
    l1 = range(10)
    l2 = [i ** 2 for i in range(10)]
    df = pd.DataFrame({"0": l1, "1": l2})
    res = ema_std(df, span=2).dropna()
    assert all((abs(dforig - res) <= 1e-15).all())


def test_compute_ma():
    stock_orig = [
        100.0,
        0.1531138587991997,
        0.6937500710898674,
        -0.9998892390840102,
        -0.46790785174554383,
        0.24992469198859263,
        0.8371986752411684,
        0.9996789142433975,
    ]
    ma10d_orig = [
        91.0,
        0.12982588130881456,
        0.6364686654113839,
        -0.9111588177100766,
        -0.45638926346295605,
        0.22777211487458476,
        0.7335679046265856,
        0.9544686462652832,
    ]
    ma30d_orig = [
        71.0,
        0.029843302068984976,
        0.40788852852895285,
        -0.5654851211095089,
        -0.3735139378462722,
        0.0648917102227224,
        0.4075702001405792,
        0.5942823972334838,
    ]
    index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    dforig = pd.DataFrame(
        {"Stock": stock_orig, "10d": ma10d_orig, "30d": ma30d_orig}, index=index
    )
    x = np.sin(np.linspace(1, 10, 100))
    df = pd.DataFrame({"Stock": x})
    ma = compute_ma(df, ema, spans=[10, 30])
    assert all(abs((dforig - ma.describe()) <= 1e-15).all())


def test_plot_bollinger_band():
    labels_orig = ["Stock", "15d", "Bollinger Band"]
    xlabel_orig = "Days"
    ylabel_orig = "Price"
    title_orig = (
        "Bollinger Band of +/- 2$\\sigma$, Moving Average " "of sma over 15 days"
    )
    x = np.sin(np.linspace(1, 10, 100))
    df = pd.DataFrame({"Stock": x}, index=np.linspace(1, 10, 100))
    df.index.name = "Days"
    plt.figure()
    plot_bollinger_band(df, sma, span=15)
    # get data from axis object
    ax = plt.gca()
    # ax.lines[0] is the data we passed to plot_bollinger_band
    # ax.lines[1] is the moving average (tested already)
    # not sure how to obtain the data of the BollingerBand from
    # the plot.
    # only checking if input data matches data of first line on plot,
    # as a measure of data appearing in the plot
    line1 = ax.lines[0]
    stock_plot = line1.get_xydata()
    labels_plot = ax.get_legend_handles_labels()[1]
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    title_plot = ax.get_title()
    # tests
    assert (df["Stock"].index.values == stock_plot[:, 0]).all()
    assert (df["Stock"].values == stock_plot[:, 1]).all()
    assert labels_orig == labels_plot
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot
    assert title_orig == title_plot
