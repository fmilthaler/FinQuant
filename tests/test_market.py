###################
# tests for Market #
###################

import numpy as np
import pandas as pd
import pytest
import yfinance

from finquant.market import Market
from finquant.portfolio import build_portfolio

d = {
    0: {"Name": "GOOG", "Allocation": 20},
    1: {"Name": "AMZN", "Allocation": 10},
    2: {"Name": "MCD", "Allocation": 15},
    3: {"Name": "DIS", "Allocation": 18},
    4: {"Name": "TSLA", "Allocation": 48},
}


pf_allocation = pd.DataFrame.from_dict(d, orient="index")

names_yf = pf_allocation["Name"].values.tolist()

# dates can be set as datetime or string, as shown below:
start_date = "2018-01-01"
end_date = "2023-01-01"


def test_Market():
    pf = build_portfolio(
        names=names_yf,
        pf_allocation=pf_allocation,
        start_date=start_date,
        end_date=end_date,
        data_api="yfinance",
        market_index="^GSPC",
    )
    assert isinstance(pf.market_index, Market)
    assert pf.market_index.name == "^GSPC"
    assert pf.beta is not None
    assert pf.rsquared is not None
