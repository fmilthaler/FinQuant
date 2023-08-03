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
from typing import Any, List, Union

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
        data=pd.DataFrame(),
        names=["name1", "name2"],
        start_date="2023-08-01",
        freq=10,
        generic_df=[df1, df2],
    )
    """

    dtype_msg: str = " with dtype 'np.float64'"

    type_dict = {
        "data": (pd.DataFrame, "a non-empty pandas.DataFrame"),
        "pf_allocation": (pd.DataFrame, "a non-empty pd.DataFrame"),
        "means": (
            Union[np.ndarray, pd.Series],
            f"a non-empty numpy.ndarray or pandas.Series {dtype_msg}.",
        ),
        "weights": (
            Union[np.ndarray, pd.Series],
            f"a non-empty numpy.ndarray or pandas.Series {dtype_msg}.",
        ),
        "cov_matrix": (
            Union[np.ndarray, pd.DataFrame],
            f"a non-empty numpy.ndarray or pandas.DataFrame {dtype_msg}.",
        ),
        "names": (
            Union[List, np.ndarray],
            "a non-empty List[str] or numpy.ndarray[str]",
        ),
        "cols": (List, "a non-empty List[str]"),
        "start_date": (
            Union[str, datetime.datetime],
            "of type str or datetime.datetime",
        ),
        "end_date": (Union[str, datetime.datetime], "of type str or datetime.datetime"),
        "data_api": (str, "of type str"),
        "market_index": (str, "of type str"),
        "freq": ((int, np.integer), "of type integer"),
        "generic_str": (str, "of type str"),
        "generic_int": ((int, np.integer), "of type integer"),
        "generic_float": ((float, np.floating), "of type float"),
        "generic_numeric": (
            (int, np.integer, float, np.floating),
            "of type integer or float",
        ),
        "generic_df": (pd.DataFrame, "a non-empty pd.DataFrame"),
        "generic_series": (pd.Series, "a non-empty pandas.Series"),
        "generic_datetime": (
            Union[str, datetime.datetime],
            "of type str or datetime.datetime",
        ),
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
            if arg_name in ("data", "means", "weights", "cov_matrix"):
                if (
                    not isinstance(arg_values, arg_type)
                    or (
                        isinstance(arg_values, Union[np.ndarray, pd.Series])
                        and not arg_values.dtype == np.float64
                    )
                    or (
                        isinstance(arg_values, pd.DataFrame)
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

            # else (arg_name is not "names" nor "cols")
            if not isinstance(arg_values, (List, np.ndarray)):
                arg_values = [arg_values]
            for arg_value in arg_values:
                if not isinstance(arg_value, arg_type):
                    raise TypeError(
                        f"Error: {arg_name} is expected to be {expected_type}."
                    )
                if (
                    isinstance(arg_value, (pd.Series, pd.DataFrame)) and arg_value.empty
                ) or (
                    isinstance(arg_value, (List, np.ndarray)) and len(arg_value) == 0
                ):
                    raise ValueError(
                        f"Error: {arg_name} is expected to be non-empty {expected_type}."
                    )
