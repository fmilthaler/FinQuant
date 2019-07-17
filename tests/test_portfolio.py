###############################################################
# testing modules portfolio, optimisation, efficient_frontier #
# all through the interfaces in Portfolio                     #
###############################################################
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime
import quandl
import yfinance
import pytest
from finquant.portfolio import build_portfolio, Stock, Portfolio
from finquant.efficient_frontier import EfficientFrontier

# comparisons
strong_abse = 1e-15
weak_abse = 1e-8

# setting quandl api key
quandl.ApiConfig.api_key = os.getenv("QUANDLAPIKEY")

# read data from file
df_pf_path = pathlib.Path.cwd() / ".." / "data" / "ex1-portfolio.csv"
df_data_path = pathlib.Path.cwd() / ".." / "data" / "ex1-stockdata.csv"
df_pf = pd.read_csv(df_pf_path)
df_data = pd.read_csv(df_data_path, index_col="Date", parse_dates=True)
# create testing variables
names = df_pf.Name.values.tolist()
names_yf = [name.replace("WIKI/", "") for name in names]
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
    {"names": names, "pf_allocation": df_pf},
    {"names": names},
    {"names": names, "start_date": start_date, "end_date": end_date},
    {
        "names": names,
        "start_date": start_date,
        "end_date": end_date,
        "data_api": "quandl",
    },
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
    {"names": ["GE"], "pf_allocation": df_pf, "data_api": "yfinance"},
    {"names": ["WIKI/GOOG"], "data": df_data},
    {"names": ["GOOG"], "data": df_data, "data_api": "yfinance"},
    {"names": names, "start_date": start_date, "end_date": "end_date"},
    {"names": names, "start_date": start_date, "end_date": 1},
    {"names": names, "data_api": "my_api"},
    {"data": [1, 2]},
    {"data": df_data.values},
    {"data": df_data, "start_date": start_date, "end_date": end_date},
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
    assert isinstance(pf.get_stock(names[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf).all()).all()
    assert (pf.comp_weights() - weights_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_1():
    d = d_pass[1]
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
    assert isinstance(pf.get_stock(names[0]), Stock)
    assert isinstance(pf.data, pd.DataFrame)
    assert isinstance(pf.portfolio, pd.DataFrame)
    assert len(pf.stocks) == len(pf.data.columns)
    assert pf.data.columns.tolist() == names
    assert pf.data.index.name == "Date"
    assert ((pf.portfolio == df_pf2).all()).all()
    assert (pf.comp_weights() - weights_no_df_pf <= strong_abse).all()
    pf.properties()


def test_buildPF_pass_4():
    d = d_pass[4]
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


def test_buildPF_pass_5():
    d = d_pass[5]
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


def test_buildPF_pass_6():
    d = d_pass[6]
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


def test_buildPF_fail_0():
    d = d_fail[0]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_1():
    d = d_fail[1]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_2():
    d = d_fail[2]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_3():
    d = d_fail[3]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_4():
    d = d_fail[4]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_5():
    d = d_fail[5]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_6():
    d = d_fail[6]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_7():
    d = d_fail[7]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_8():
    d = d_fail[8]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_9():
    d = d_fail[9]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_10():
    d = d_fail[10]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_11():
    d = d_fail[11]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_12():
    d = d_fail[12]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_13():
    d = d_fail[13]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_14():
    d = d_fail[14]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_15():
    d = d_fail[15]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_16():
    d = d_fail[16]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_17():
    d = d_fail[17]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_18():
    d = d_fail[18]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_19():
    d = d_fail[19]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_20():
    d = d_fail[20]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_21():
    d = d_fail[21]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_22():
    d = d_fail[22]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_23():
    d = d_fail[23]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_24():
    d = d_fail[24]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_25():
    d = d_fail[25]
    with pytest.raises(Exception):
        build_portfolio(**d)


def test_buildPF_fail_26():
    d = d_fail[26]
    with pytest.raises(Exception):
        build_portfolio(**d)


###################
# tests for Stock #
###################


def test_Stock():
    d = d_pass[3]
    pf = build_portfolio(**d)
    # loop over all stocks stored within pf and check that values
    # are equal to the ones in pf
    for i in range(len(pf.stocks)):
        assert isinstance(pf.get_stock(names[0]), Stock)
        stock = pf.get_stock(names[i])
        assert stock.name == pf.portfolio["Name"][i]
        assert all(stock.data - pf.data[stock.name].to_frame() <= strong_abse)
        assert all(
            stock.investmentinfo == pf.portfolio.loc[pf.portfolio["Name"] == stock.name]
        )


######################################
# tests for Monte Carlo optimisation #
######################################


def test_mc_optimisation():
    d = d_pass[3]
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
    # also test the plot
    plt.figure()
    pf.mc_plot_results()
    # get axis object
    ax = plt.gca()
    # only checking legend and axis labels, and assume that the plot
    # was successfully created
    labels_plot = ax.get_legend_handles_labels()[1]
    xlabel_plot = ax.get_xlabel()
    ylabel_plot = ax.get_ylabel()
    assert labels_orig == labels_plot
    assert xlabel_orig == xlabel_plot
    assert ylabel_orig == ylabel_plot


#############################################
# tests for Efficient Frontier optimisation #
#############################################


def test_get_ef():
    d = d_pass[3]
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
    d = d_pass[3]
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
    d = d_pass[3]
    pf = build_portfolio(**d)
    max_sharpe_weights = np.array(
        [0.0, 0.41322217986076903, 0.5867778201392311, 2.2858514942065993e-17]
    )
    ef_opt_weights = pf.ef_maximum_sharpe_ratio()
    assert np.allclose(
        ef_opt_weights.values.transpose(), max_sharpe_weights, atol=weak_abse
    )


def test_efficient_return():
    d = d_pass[3]
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
    d = d_pass[3]
    pf = build_portfolio(**d)
    efficient_volatility_weights = np.array(
        [0.0, 0.5325563159992046, 0.4674436840007955, 0.0]
    )
    ef_opt_weights = pf.ef_efficient_volatility(0.191)
    assert np.allclose(
        ef_opt_weights.values.transpose(), efficient_volatility_weights, atol=weak_abse
    )


def test_efficient_frontier():
    d = d_pass[3]
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


def test_plot_efrontier():
    d = d_pass[3]
    pf = build_portfolio(**d)
    # just checking if a plot is successfully created,
    # not checking for content
    pf.ef_plot_efrontier()


def test_plot_optimal_portfolios():
    d = d_pass[3]
    pf = build_portfolio(**d)
    # just checking if a plot is successfully created,
    # not checking for content
    pf.ef_plot_optimal_portfolios()


def test_plot_stocks():
    d = d_pass[3]
    pf = build_portfolio(**d)
    # just checking if a plot is successfully created, not checking for content
    pf.plot_stocks()
