"""
type_utilities.py

This module defines a type validation utility for working with various data types in Python, utilizing the 'numpy'
and 'pandas' libraries.

Dependencies:
-------------
This module requires the following external libraries:

- 'numpy' (imported as 'np')
- 'pandas' (imported as 'pd')

Example usage:
--------------
Example:

.. code-block:: python

    type_validation(
        data=pd.DataFrame([1., 2.]),
        names=["name1", "name2"],
        start_date="2023-08-01",
        freq=10.0,
    )

"""

import datetime
from typing import Any, Callable, List, Type

import numpy as np
import pandas as pd


# Arrays, Series, DataFrames:
def _check_series_or_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = f"a non-empty pandas.Series or pandas.DataFrame with dtype '{element_type.__name__}'."
    if not isinstance(arg_values, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, pd.Series) and not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, pd.DataFrame) and not all(
        arg_values.dtypes == element_type
    ):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if arg_values.empty:
        raise ValueError(
            f"Error: {arg_name} is expected to be non-empty {expected_type_msg}."
        )


def _check_dataframe_any_subtype(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, pd.DataFrame):
        raise TypeError(
            f"Error: {arg_name} is expected to be a non-empty pandas.DataFrame."
        )
    if arg_values.empty:
        raise ValueError(
            f"Error: {arg_name} is expected to be a non-empty pandas.DataFrame."
        )


def _check_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = (
        f"a non-empty pandas.DataFrame with dtype '{element_type.__name__}'."
    )
    if not isinstance(arg_values, pd.DataFrame) or not all(
        arg_values.dtypes == element_type
    ):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if arg_values.empty:
        raise ValueError(
            f"Error: {arg_name} is expected to be non-empty {expected_type_msg}."
        )


def _check_series_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = (
        f"a non-empty pandas.Series with dtype '{element_type.__name__}'."
    )
    if not isinstance(arg_values, pd.Series) or not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if arg_values.empty:
        raise ValueError(
            f"Error: {arg_name} is expected to be non-empty {expected_type_msg}."
        )


def _check_array_or_series_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = f"a non-empty numpy.ndarray or pandas.Series with dtype '{element_type.__name__}'."
    if not isinstance(arg_values, (np.ndarray, pd.Series)):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, pd.Series) and not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be {expected_type_msg}.")


def _check_array_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = (
        f"a non-empty numpy.ndarray with dtype '{element_type.__name__}'."
    )
    if not isinstance(arg_values, np.ndarray) or not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be {expected_type_msg}.")


def _check_array_or_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type = np.float64
) -> None:
    expected_type_msg = f"a non-empty numpy.ndarray or pandas.DataFrame with dtype '{element_type.__name__}'."
    if not isinstance(arg_values, (np.ndarray, pd.DataFrame)):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, np.ndarray) and not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, pd.DataFrame) and not all(
        arg_values.dtypes == element_type
    ):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be {expected_type_msg}.")


# Lists or Arrays:
def _check_list_or_array_str(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (List, np.ndarray)) or not all(
        isinstance(val, str) for val in arg_values
    ):
        raise TypeError(
            f"Error: {arg_name} is expected to be a non-empty List[str] or numpy.ndarray[str]."
        )
    if len(arg_values) == 0:
        raise ValueError(
            f"Error: {arg_name} is expected to be a non-empty List[str] or numpy.ndarray[str]."
        )


def _check_list_or_array_int(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (List, np.ndarray)) or not all(
        isinstance(val, (int, np.integer, float, np.floating)) for val in arg_values
    ):
        raise TypeError(
            f"Error: {arg_name} is expected to be a non-empty List or numpy.ndarray of int or float."
        )
    if len(arg_values) == 0:
        raise ValueError(
            f"Error: {arg_name} is expected to be a non-empty List or numpy.ndarray of int or float."
        )


def _check_list_int(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, List) or not all(
        isinstance(val, int) for val in arg_values
    ):
        raise TypeError(f"Error: {arg_name} is expected to be a non-empty List[int].")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be a non-empty List[int].")


# Datetime types:
def _check_datetime_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (str, datetime.datetime)):
        raise TypeError(
            f"Error: {arg_name} is expected to be of type str or datetime.datetime."
        )


# Numberic types:
def _check_float_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (float, np.floating)):
        raise TypeError(f"Error: {arg_name} is expected to be of type float.")


def _check_integer_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (int, np.integer)):
        raise TypeError(f"Error: {arg_name} is expected to be of type integer.")


def _check_numeric_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (int, np.integer, float, np.floating)):
        raise TypeError(
            f"Error: {arg_name} is expected to be of type integer or float."
        )


# Boolean types:
def _check_bool_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, bool):
        raise TypeError(f"Error: {arg_name} is expected to be of type bool.")


# String types:
def _check_string_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, str):
        raise TypeError(f"Error: {arg_name} is expected to be of type str.")


# Callable types:
def _check_callable_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, Callable):
        raise TypeError(f"Error: {arg_name} is expected to be a Callable function.")


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

    # Definition of potential arguments and corresponding expected types
    type_dict = {
        # DataFrames, Series, Arrays:
        "data": _check_series_or_dataframe_float,
        "pf_allocation": _check_dataframe_any_subtype,
        "returns_df": _check_dataframe_float,
        "returns_series": _check_series_float,
        "market_daily_returns": _check_series_float,
        "means": _check_array_or_series_float,
        "weights": _check_array_or_series_float,
        "initial_weights": _check_array_float,
        "weights_array": _check_array_float,
        "cov_matrix": _check_array_or_dataframe_float,
        # Lists:
        "names": _check_list_or_array_str,
        "cols": _check_list_or_array_str,
        "spans": _check_list_int,
        "targets": _check_list_or_array_int,
        # Datetime objects:
        "start_date": _check_datetime_type,
        "end_date": _check_datetime_type,
        # Strings:
        "data_api": _check_string_type,
        "market_index": _check_string_type,
        "method": _check_string_type,
        "name": _check_string_type,
        # FLOATs
        "expected_return": _check_float_type,
        "volatility": _check_float_type,
        "risk_free_rate": _check_float_type,
        "downside_risk": _check_float_type,
        "mu": _check_float_type,
        "sigma": _check_float_type,
        "conf_level": _check_float_type,
        # INTs:
        "freq": _check_integer_type,
        "span": _check_integer_type,
        "num_trials": _check_integer_type,
        # NUMERICs:
        "investment": _check_numeric_type,
        "dividend": _check_numeric_type,
        "target": _check_numeric_type,
        # Booleans:
        "plot": _check_bool_type,
        "save_weights": _check_bool_type,
        "verbose": _check_bool_type,
        "defer_update": _check_bool_type,
        # Callables:
        "fun": _check_callable_type,
    }

    for arg_name, arg_values in kwargs.items():
        if arg_name not in type_dict:
            raise ValueError(
                f"Error: '{arg_name}' is not a valid argument. Please only use argument names defined in `type_dict`."
            )

        # Some arguments are allowed to be None, so skip them
        if arg_values is None:
            continue

        # Extract the appropriate validation function from the type_dict
        validation_func = type_dict[arg_name]

        # Perform the type validation using the appropriate function
        validation_func(arg_name, arg_values)
