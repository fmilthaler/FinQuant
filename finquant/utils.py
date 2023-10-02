import datetime
from typing import Optional
import pandas as pd

from finquant.data_types import ELEMENT_TYPE, LIST_DICT_KEYS, SERIES_OR_DATAFRAME, INT
from finquant.portfolio import _yfinance_request
from finquant.type_utilities import type_validation


def all_list_ele_in_other(
    l_1: LIST_DICT_KEYS[ELEMENT_TYPE],
    l_2: LIST_DICT_KEYS[ELEMENT_TYPE],
) -> bool:
    """Returns True if all elements of list l1 are found in list l2."""
    return all(ele in l_2 for ele in l_1)


def re_download_stock_data(data: SERIES_OR_DATAFRAME, stock_name: str, num_days_predate_stock_price: Optional[INT] = 0) -> pd.DataFrame:
    # Type validations:
    type_validation(data=data, name=stock_name, num_days_predate_stock_price=num_days_predate_stock_price)
    if num_days_predate_stock_price < 0:
        raise ValueError("Error: num_days_predate_stock_price must be >= 0.")
    # download additional price data 'Open' for given stock and timeframe:
    start_date = data.index.min() - datetime.timedelta(days=num_days_predate_stock_price)
    end_date = data.index.max() + datetime.timedelta(days=1)
    df = _yfinance_request([stock_name], start_date=start_date, end_date=end_date)
    # dropping second level of column header that yfinance returns
    df.columns = df.columns.droplevel(1)
    return df
