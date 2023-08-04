"""
type_validation.py

This module defines a type validation utility for working with various data types in Python, utilizing the 'numpy'
and 'pandas' libraries.

Dependencies:
-------------
This module requires the following external libraries:

- 'numpy' (imported as 'np')
- 'pandas' (imported as 'pd')

Example usage:
-------------
type_validation(
    data=pd.DataFrame([1., 2.]),
    names=["name1", "name2"],
    start_date="2023-08-01",
    freq=10.0,
)
"""
# allow more than 5 boolean expressions in if statement:
# pylint: disable=R0916


import datetime
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd


def type_validation(**kwargs: Any) -> None:
    """
    Perform generic type validations on input variables.

    This function performs various type validations on a set of input variables. It helps to ensure that the input
    values conform to the expected types and conditions, raising a TypeError with a descriptive error message
    if any type validation fails and a ValueError if a numpy.array or pd.Series/DataFrame is empty.

    Parameters:
        **kwargs (Any): Arbitrary keyword arguments representing the input variables to be checked.

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

    # Definition of common types to check against with error message:
    dataframe_any_type: Tuple[Any, str] = (pd.DataFrame, "a non-empty pandas.DataFrame")
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
    list_int_type: Tuple[Any, str] = (List, "a non-empty List[int]")
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
    bool_type: Tuple[Any, str] = (bool, "of type bool")
    callable_type: Tuple[Any, str] = (Callable, "a Callable function")

    # Definition of potential arguments and corresponding expected types
    type_dict = {
        # DataFrames, Series, Arrays:
        "data": series_dataframe_float_type,
        "pf_allocation": dataframe_any_type,  # allows for any subtype
        "means": array_series_floats_type,
        "weights": array_series_floats_type,
        "cov_matrix": array_dataframe_floats_type,
        # Lists:
        "names": list_array_type,
        "cols": list_array_type,
        "spans": list_int_type,
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
        "span": int_type,
        # NUMERICs:
        "investment": numeric_type,
        "dividend": numeric_type,
        # Booleans:
        "plot": bool_type,
        # Callables:
        "fun": callable_type,
    }

    # Type validation
    for arg_name, (arg_type, expected_type) in type_dict.items():
        arg_values = kwargs.get(arg_name)
        if arg_values is not None:
            # Validating List[str] types:
            if arg_name in ("names", "cols", "spans"):
                # spans is expected to be List[INT], the rest List[str]
                if not isinstance(arg_values, arg_type) or (
                    isinstance(arg_values, arg_type)
                    and (
                        arg_name == "spans"
                        and not all(
                            isinstance(val, (np.integer, int)) for val in arg_values
                        )
                        or (
                            arg_name != "spans"
                            and not all(isinstance(val, str) for val in arg_values)
                        )
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

            # Validating common Array[FLOAT], Series[Float], DataFrame[Any] types
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
