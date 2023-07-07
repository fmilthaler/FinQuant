###################
# tests for Stock #
###################

import os
import pathlib
import numpy as np
import pandas as pd
import datetime
import quandl
import yfinance
import pytest
from finquant.portfolio import build_portfolio, Stock

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

# create kwargs to be passed to build_portfolio
d_pass = [
    {
        "names": names,
        "start_date": start_date,
        "end_date": end_date,
        "data_api": "quandl",
    }
]


def test_Stock():
    d = d_pass[0]
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
