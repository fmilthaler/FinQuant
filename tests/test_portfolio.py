###############################################################
# testing modules portfolio, optimisation, efficient_frontier #
# all through the interfaces in Portfolio                     #
###############################################################

import datetime
import os
import pathlib

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pytest
import quandl
import yfinance

from finquant.efficient_frontier import EfficientFrontier
from finquant.portfolio import Portfolio, build_portfolio
from finquant.stock import Stock

# comparisons
strong_abse = 1e-15
weak_abse = 1e-8

# setting quandl api key
quandl.ApiConfig.api_key = os.getenv("QUANDLAPIKEY")
quandl.read_key()

# read data from file
df_pf_path = pathlib.Path.cwd() / ".." / "data" / "ex1-portfolio.csv"
df_data_path = pathlib.Path.cwd() / ".." / "data" / "ex1-stockdata.csv"
# allocation of portfolio (quandl version):
df_pf = pd.read_csv(df_pf_path)
# allocation of portfolio (yfinance version):
df_pf_yf = df_pf.copy()
df_pf_yf["Name"] = df_pf_yf["Name"].str.replace("WIKI/", "")
# stock price data (quandl version):
df_data = pd.read_csv(df_data_path, index_col="Date", parse_dates=True)
# stock price data (yfinance version):
df_data_yf = df_data.copy()
df_data_yf = df_data_yf.rename(columns=lambda x: x.replace("WIKI/", ""))
# create testing variables
names = df_pf.Name.values.tolist()
names_yf = df_pf_yf.Name.values.tolist()
# weights
weights_df_pf = [
    0.31746031746031744,
    0.15873015873015872,
    0.23809523809523808,
    0.2857142857142857,
]
weights_no_df_pf = [1.0 / len(names) for i in range(len(names))]
df_pf2 = pd.DataFrame({"Allocation": weights_no_df_pf, "Name": names})
df_pf2_yf = pd.DataFrame({"Allocation": weights_no_df_pf, "Name": names_yf})
start_date = datetime.datetime(2015, 1, 1)
end_date = "2017-12-31"
# portfolio quantities (based on provided data)
expret_orig = 0.2382653706795801
vol_orig = 0.1498939453149472
sharpe_orig = 1.5562027551510393
freq_orig = 252
risk_free_rate_orig = 0.005
# create fake allocations
d_error_1 = {
    0: {"Names": "WIKI/GOOG", "Allocation": 20},
    1: {"Names": "WIKI/AMZN", "Allocation": 10},
    2: {"Names": "WIKI/MCD", "Allocation": 15},
    3: {"Names": "WIKI/DIS", "Allocation": 18},
}
df_pf_error_1 = pd.DataFrame.from_dict(d_error_1, orient="index")
d_error_2 = {
    0: {"Name": "WIKI/GOOG", "weight": 20},
    1: {"Name": "WIKI/AMZN", "weight": 10},
    2: {"Name": "WIKI/MCD", "weight": 15},
    3: {"Name": "WIKI/DIS", "weight": 18},
}
df_pf_error_2 = pd.DataFrame.from_dict(d_error_2, orient="index")
d_error_3 = {
    0: {"Name": "WIKI/IBM", "Allocation": 20},
    1: {"Name": "WIKI/KO", "Allocation": 10},
    2: {"Name": "WIKI/AXP", "Allocation": 15},
    3: {"Name": "WIKI/GE", "Allocation": 18},
}
df_pf_error_3 = pd.DataFrame.from_dict(d_error_3, orient="index")
d_error_4 = {
    0: {"Name": "WIKI/GOOG", "Allocation": 20},
    1: {"Name": "WIKI/AMZN", "Allocation": 10},
    2: {"Name": "WIKI/MCD", "Allocation": 15},
    3: {"Name": "WIKI/GE", "Allocation": 18},
}
df_pf_error_4 = pd.DataFrame.from_dict(d_error_4, orient="index")
# create kwargs to be passed to build_portfolio
d_pass = [
    {"names": names_yf, "pf_allocation": df_pf_yf, "data_api": "yfinance"},
    {"names": names_yf, "data_api": "yfinance"},
    {
        "names": names,
        "start_date": start_date,
        "end_date": end_date,
    },  # testing default (quandl)
    {
        "names": names_yf,
        "start_date": start_date,
        "end_date": end_date,
        "data_api": "yfinance",
    },
    {"data": df_data},
    {"data": df_data, "pf_allocation": df_pf},
]
d_fail = [
    {},
    {"testinput": "..."},
    {"names": ["WIKI/GOOG"], "testinput": "..."},
    {"names": 1},
    {"names": "test"},
    {"names": "test", "data_api": "yfinance"},
    {"names": "WIKI/GOOG"},
    {"names": "GOOG", "data_api": "yfinance"},
    {"names": ["WIKI/GE"], "pf_allocation": df_pf},
    {"names": ["GE"], "pf_allocation": df_pf_yf, "data_api": "yfinance"},
    {"names": ["WIKI/GOOG"], "data": df_data},
    {"names": ["GOOG"], "data": df_data_yf, "data_api": "yfinance"},
    {"names": names, "start_date": start_date, "end_date": "end_date"},
    {"names": names, "start_date": start_date, "end_date": 1},
    {"names": names, "data_api": "my_api"},
    {"data": [1, 2]},
    {"data": df_data.values},
    {"data": df_data, "start_date": start_date, "end_date": end_date},
    {"data": df_data, "data_api": "quandl"},
    {"data": df_data, "data_api": "yfinance"},
    {"data": df_data, "pf_allocation": df_data},
    {"data": df_pf, "pf_allocation": df_pf},
    {"data": df_data, "pf_allocation": df_pf_error_1},
    {"data": df_data, "pf_allocation": df_pf_error_2},
    {"data": df_data, "pf_allocation": df_pf_error_3},
    {"data": df_data, "pf_allocation": df_pf_error_4},
    {"data": df_data, "pf_allocation": "test"},
    {"pf_allocation": df_pf},
]


#################################################
# tests that are meant to successfully build pf #
#################################################


def test_buildPF_pass_0():
    d = d_pass[0]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.get_stock(names_yf[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names_yf
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf_yf).all()).all()
    assert (pf.comp_weights() - weights_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_1():
    d = d_pass[1]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.get_stock(names_yf[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names_yf
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf2_yf).all()).all()
    assert (pf.comp_weights() - weights_no_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_2():
    d = d_pass[2]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.get_stock(names[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf2).all()).all()
    assert (pf.comp_weights() - weights_no_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_3():
    d = d_pass[3]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.get_stock(names_yf[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names_yf
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf2_yf).all()).all()
    assert (pf.comp_weights() - weights_no_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_4():
    d = d_pass[4]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.get_stock(names[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf2).all()).all()
    assert (pf.comp_weights() - weights_no_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_5():
    d = d_pass[5]
    pf = build_portfolio(**d)
    assert isinstance(pf, Portfolio)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf).all()).all()
    assert (pf.comp_weights() - weights_df_pf <= strong_abse).all()
    assert expret_orig - pf.expected_return <= strong_abse
    assert vol_orig - pf.volatility <= strong_abse
    assert sharpe_orig - pf.sharpe <= strong_abse
    assert freq_orig - pf.freq <= strong_abse
    assert risk_free_rate_orig - pf.risk_free_rate <= strong_abse
    pf.properties()


#####################################################################
# tests that are meant to raise an exception during build_portfolio #
#####################################################################


def test_expected_fails():
    for fail_test_settings in d_fail:
        print("++++++++++++++++++++++++++++++++++++")
        print("fail_test_settings: ", fail_test_settings)
        with pytest.raises(Exception):
            build_portfolio(**fail_test_settings)


######################################
# tests for Monte Carlo optimisation #
######################################


def test_mc_optimisation():
    d = d_pass[4]
    pf = build_portfolio(**d)
    # since the monte carlo optimisation is based on random numbers,
    # we set a seed, so that the results can be compared.
    np.random.seed(seed=0)
    # orig values:
    minvol_res_orig = [0.18560926749041448, 0.1333176229402258, 1.3547291311321408]
    maxsharpe_res_orig = [0.33033770744503416, 0.16741461860370618, 1.9433052511092475]
    minvol_w_orig = [
        0.09024151309669741,
        0.015766238378839476,
        0.514540537132381,
        0.37945171139208195,
    ]
    maxsharpe_w_orig = [
        0.018038812127367274,
        0.37348740385059126,
        0.5759648343129179,
        0.03250894970912355,
    ]
    labels_orig = ["min Volatility", "max Sharpe Ratio", "Initial Portfolio"]
    xlabel_orig = "Volatility [period=252]"
    ylabel_orig = "Expected Return [period=252]"
    # run Monte Carlo optimisation through pf
    opt_w, opt_res = pf.mc_optimisation(num_trials=500)
    # tests
    assert (minvol_res_orig - opt_res.iloc[0].values <= strong_abse).all()
    assert (maxsharpe_res_orig - opt_res.iloc[1].values <= strong_abse).all()
    assert (minvol_w_orig - opt_w.iloc[0].values <= strong_abse).all()
    assert (maxsharpe_w_orig - opt_w.iloc[1].values <= strong_abse).all()


#############################################
# tests for Efficient Frontier optimisation #
#############################################


def test_get_ef():
    d = d_pass[4]
    pf = build_portfolio(**d)
    ef = pf._get_ef()
    assert isinstance(ef, EfficientFrontier)
    assert isinstance(pf.ef, EfficientFrontier)
    assert pf.ef == ef
    assert (pf.comp_mean_returns(freq=1) == ef.mean_returns).all()
    assert (pf.comp_cov() == ef.cov_matrix).all().all()
    assert pf.freq == ef.freq
    assert pf.risk_free_rate == ef.risk_free_rate
    assert ef.names == pf.portfolio["Name"].values.tolist()
    assert ef.num_stocks == len(pf.stocks)


def test_ef_minimum_volatility():
    d = d_pass[4]
    pf = build_portfolio(**d)
    min_vol_weights = np.array(
        [
            0.15515521225480033,
            2.168404344971009e-18,
            0.4946241856546514,
            0.35022060209054834,
        ]
    )
    ef_opt_weights = pf.ef_minimum_volatility()
    assert np.allclose(
        ef_opt_weights.values.transpose(), min_vol_weights, atol=weak_abse
    )


def test_maximum_sharpe_ratio():
    d = d_pass[4]
    pf = build_portfolio(**d)
    max_sharpe_weights = np.array(
        [0.0, 0.41322217986076903, 0.5867778201392311, 2.2858514942065993e-17]
    )
    ef_opt_weights = pf.ef_maximum_sharpe_ratio()
    assert np.allclose(
        ef_opt_weights.values.transpose(), max_sharpe_weights, atol=weak_abse
    )


def test_efficient_return():
    d = d_pass[4]
    pf = build_portfolio(**d)
    efficient_return_weights = np.array(
        [
            0.09339785373366818,
            0.1610699937778475,
            0.5623652130240258,
            0.18316693946445858,
        ]
    )
    ef_opt_weights = pf.ef_efficient_return(0.2542)
    assert np.allclose(
        ef_opt_weights.values.transpose(), efficient_return_weights, atol=weak_abse
    )


def test_efficient_volatility():
    d = d_pass[4]
    pf = build_portfolio(**d)
    efficient_volatility_weights = np.array(
        [0.0, 0.5325563159992046, 0.4674436840007955, 0.0]
    )
    ef_opt_weights = pf.ef_efficient_volatility(0.191)
    assert np.allclose(
        ef_opt_weights.values.transpose(), efficient_volatility_weights, atol=weak_abse
    )


def test_efficient_frontier():
    d = d_pass[4]
    pf = build_portfolio(**d)
    efrontier = np.array(
        [
            [0.13309804281267365, 0.2],
            [0.13382500317474913, 0.21],
            [0.13491049715255884, 0.22],
            [0.13634357140158387, 0.23],
            [0.13811756026939967, 0.24],
            [0.14021770106367654, 0.25],
            [0.1426296319926618, 0.26],
            [0.1453376889002686, 0.27],
            [0.14832574432346657, 0.28],
            [0.15157724434785497, 0.29],
            [0.15507572162309227, 0.3],
        ]
    )
    targets = [round(0.2 + 0.01 * i, 2) for i in range(11)]
    ef_efrontier = pf.ef_efficient_frontier(targets)
    assert np.allclose(ef_efrontier, efrontier, atol=weak_abse)


########################################################
# tests for some portfolio/efficient frontier plotting #
# only checking for errors/exceptions that pop up      #
########################################################
plt.switch_backend("Agg")


def test_plot_efrontier():
    d = d_pass[4]
    pf = build_portfolio(**d)
    # Clear the current figure to ensure a fresh plot
    plt.clf()
    # Create the plot
    pf.ef_plot_efrontier()
    # Assert that the plot was created
    assert len(plt.gcf().get_axes()) == 1

    # get title, axis labels and axis min/max values
    ax = plt.gca()
    title = ax.get_title()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # expected values:
    expected_title = "Efficient Frontier"
    expected_xlable = "Volatility"
    expected_ylable = "Expected Return"
    expected_xlim = (0.12504, 0.29359)
    expected_ylim = (0.05445, 0.50655)

    # assert on title, labels and limits
    assert title == expected_title
    assert xlabel == expected_xlable
    assert ylabel == expected_ylable
    assert xlim == pytest.approx(expected_xlim, 1e-3)
    assert ylim == pytest.approx(expected_ylim, 1e-3)


def test_plot_optimal_portfolios():
    d = d_pass[4]
    pf = build_portfolio(**d)
    # Clear the current figure to ensure a fresh plot
    plt.clf()
    # Create the plot
    pf.ef_plot_optimal_portfolios()
    # Assert that the plot was created
    assert len(plt.gcf().get_axes()) == 1

    # get axis min/max values
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # expected values:
    expected_xlim = (0.13064, 0.17607)
    expected_ylim = (0.17957, 0.35317)

    # assert on title, labels and limits
    assert xlim == pytest.approx(expected_xlim, 1e-3)
    assert ylim == pytest.approx(expected_ylim, 1e-3)


def test_plot_stocks():
    d = d_pass[4]
    pf = build_portfolio(**d)
    # Clear the current figure to ensure a fresh plot
    plt.clf()
    # Create the plot
    pf.plot_stocks()
    # Assert that the plot was created
    assert len(plt.gcf().get_axes()) == 1

    # get axis min/max values
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # expected values:
    expected_xlim = (0.15426, 0.29255)
    expected_ylim = (0.05403, 0.50693)

    # assert on title, labels and limits
    assert xlim == pytest.approx(expected_xlim, 1e-3)
    assert ylim == pytest.approx(expected_ylim, 1e-3)


########################################################
# tests for some portfolio clustering should fail      #
########################################################

def test_cluster_stocks_fail_0():
    d = d_pass[4]
    pf = build_portfolio(**d)
    with pytest.raises(Exception):
        pf.cluster_stocks(n_clusters=5)


def test_cluster_stocks_fail_1():
    d = d_pass[4]
    pf = build_portfolio(**d)
    with pytest.raises(Exception):
        pf.cluster_stocks(n_clusters=1)

########################################################
# tests for some portfolio clustering should pass      #
########################################################

def test_cluster_stocks_pass_0():
    d = d_pass[4]
    pf = build_portfolio(**d)
    pf.cluster_stocks(n_clusters=4, verbose=True)
