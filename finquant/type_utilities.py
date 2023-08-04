"""
This module defines type validation utility for working with various data types in Python,
utilizing the 'numpy' and 'pandas' libraries.

Dependencies:
-------------
This module requires the following external libraries:

- 'numpy' (imported as 'np')
- 'pandas' (imported as 'pd')
"""

import datetime
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


def type_validation(**kwargs: Any) -> None:
    """
    Perform generic type validations on input variables.

    This function performs various type validations on a set of input variables. It helps to ensure that the input
    values conform to the expected types and conditions, raising a TypeError with a descriptive error message
    if any type validation fails and a ValueError if a numpy.array or pd.Series/DataFrame is empty.

    Parameters:
        **kwargs: Arbitrary keyword arguments representing the input variables to be checked.

    Raises:
        TypeError: If any of the type validations fail, a TypeError is raised with a descriptive error message
                   indicating the expected type and conditions for each variable.
        ValueError: If any of the value validations fail, a ValueError is raised with a descriptive error message
                    indicating the expected conditions for each variable.

    Example usage:
        type_validation(
        data=pd.DataFrame([1., 2.]),
        names=["name1", "name2"],
        start_date="2023-08-01",
        freq=10.0,
    )
    """

    dtype_msg: str = " with dtype 'np.float64'"

    dataframe_floats_type: Tuple[Any, str] = (
        pd.DataFrame,
        f"a non-empty pandas.DataFrame {dtype_msg}",
    )
    dataframe_type: Tuple[Any, str] = (pd.DataFrame, f"a non-empty pandas.DataFrame")
    series_dataframe_float_type: Tuple[Any, str] = (
        Union[pd.Series, pd.DataFrame],
        f"a non-empty pandas.Series or pandas.DataFrame {dtype_msg}",
    )
    array_series_floats_type: Tuple[Any, str] = (
        Union[np.ndarray, pd.Series],
        f"a non-empty numpy.ndarray or pandas.Series {dtype_msg}.",
    )
    array_dataframe_floats_type: Tuple[Any, str] = (
        Union[np.ndarray, pd.DataFrame],
        f"a non-empty numpy.ndarray or pandas.DataFrame {dtype_msg}.",
    )
    list_array_type: Tuple[Any, str] = (
        Union[List, np.ndarray],
        "a non-empty List[str] or numpy.ndarray[str]",
    )
    datetime_type: Tuple[Any, str] = (
        Union[str, datetime.datetime],
        "of type str or datetime.datetime",
    )
    string_type: Tuple[Any, str] = (str, "of type str")
    float_type: Tuple[Any, str] = ((float, np.floating), "of type float")
    int_type: Tuple[Any, str] = ((int, np.integer), "of type integer")
    numeric_type: Tuple[Any, str] = (
        (int, np.integer, float, np.floating),
        "of type integer of float",
    )

    type_dict = {
        # DataFrames, Series, Arrays:
        "data": series_dataframe_float_type,
        "pf_allocation": dataframe_type,  # allows for any subtype
        "means": array_series_floats_type,
        "weights": array_series_floats_type,
        "cov_matrix": array_dataframe_floats_type,
        # Lists:
        "names": list_array_type,
        "cols": list_array_type,
        # Datatime objects:
        "start_date": datetime_type,
        "end_date": datetime_type,
        # Strings:
        "data_api": string_type,
        "market_index": string_type,
        # FLOATs
        "expected_return": float_type,
        "volatility": float_type,
        "risk_free_rate": float_type,
        "downside_risk": float_type,
        "mu": float_type,
        "sigma": float_type,
        "conf_level": float_type,
        # INTs:
        "freq": int_type,
        # NUMERICs:
        "investment": numeric_type,
        "dividend": numeric_type,
    }

    for arg_name, (arg_type, expected_type) in type_dict.items():
        arg_values = kwargs.get(arg_name)
        if arg_values is not None:
            # Validating List[str] types:
            if arg_name in ("names", "cols"):
                if not isinstance(arg_values, arg_type) or (
                    isinstance(arg_values, arg_type)
                    and not all(isinstance(val, str) for val in arg_values)
                ):
                    raise TypeError(
                        f"Error: {arg_name} is expected to be {expected_type}."
                    )
                if len(arg_values) == 0:
                    raise ValueError(
                        f"Error: {arg_name} is expected to be {expected_type}."
                    )
                continue

            # Validating common Array[FLOAT], Series[Float], DataFrame[FLOAT] types
            if arg_name in ("data", "pf_allocation", "means", "weights", "cov_matrix"):
                if (
                    not isinstance(arg_values, arg_type)
                    or (
                        isinstance(arg_values, (np.ndarray, pd.Series))
                        and not arg_values.dtype == np.float64
                    )
                    or (
                        isinstance(arg_values, pd.DataFrame)
                        and arg_name != "pf_allocation"  # allows for any subtypes
                        and not all(arg_values.dtypes == np.float64)
                    )
                ):
                    raise TypeError(
                        f"Error: {arg_name} is expected to be {expected_type}."
                    )
                if len(arg_values) == 0:
                    raise ValueError(
                        f"Error: {arg_name} is expected to be {expected_type}."
                    )
                continue

            # Remaining types:
            if not isinstance(arg_values, arg_type):
                raise TypeError(f"Error: {arg_name} is expected to be {expected_type}.")
            if (
                isinstance(arg_values, (pd.Series, pd.DataFrame)) and arg_values.empty
            ) or (isinstance(arg_values, (List, np.ndarray)) and len(arg_values) == 0):
                raise ValueError(
                    f"Error: {arg_name} is expected to be non-empty {expected_type}."
                )
