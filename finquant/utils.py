import datetime

import pandas as pd

from finquant.data_types import ELEMENT_TYPE, LIST_DICT_KEYS, SERIES_OR_DATAFRAME
from finquant.portfolio import _yfinance_request
from finquant.type_utilities import type_validation


def all_list_ele_in_other(
    l_1: LIST_DICT_KEYS[ELEMENT_TYPE],
    l_2: LIST_DICT_KEYS[ELEMENT_TYPE],
) -> bool:
    """Returns True if all elements of list l1 are found in list l2."""
    return all(ele in l_2 for ele in l_1)


def re_download_stock_data(
    data: SERIES_OR_DATAFRAME,
    stock_name: str
) -> pd.DataFrame:
    # Type validations:
    type_validation(
        data=data,
        name=stock_name,
    )
    # download additional price data 'Open' for given stock and timeframe:
    start_date = data.index.min() - datetime.timedelta(days=31)
    end_date = data.index.max() + datetime.timedelta(days=1)
    df = _yfinance_request([stock_name], start_date=start_date, end_date=end_date)
    # dropping second level of column header that yfinance returns
    df.columns = df.columns.droplevel(1)
    return df
