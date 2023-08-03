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
    Perform generic type checks on input variables.

    This function performs various type checks on a set of input variables. It helps to ensure that the input
    values conform to the expected types and conditions, raising a TypeError with a descriptive error message
    if any type check fails and a ValueError if a np.array or pd.Series/DataFrame is empty.

    Parameters:
        **kwargs: Arbitrary keyword arguments representing the input variables to be checked.

    Raises:
        TypeError: If any of the type checks fail, a TypeError is raised with a descriptive error message
                   indicating the expected type and conditions for each variable.
        ValueError: If any of the value checks fail, a ValueError is raised with a descriptive error message
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

    type_dict = {
        "data": (pd.DataFrame, "a non-empty pandas.DataFrame"),
        "pf_allocation": (pd.DataFrame, "a non-empty pd.DataFrame"),
        "names": (Union[List, np.ndarray], "a non-empty List[str] or np.ndarray[str]"),
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
