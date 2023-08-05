import datetime
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd


# Arrays, Series, DataFrames:
def check_series_or_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
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


def check_dataframe_any_subtype(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, pd.DataFrame):
        raise TypeError(
            f"Error: {arg_name} is expected to be a non-empty pandas.DataFrame."
        )
    if arg_values.empty:
        raise ValueError(
            f"Error: {arg_name} is expected to be a non-empty pandas.DataFrame."
        )


def check_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
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


def check_series_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
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


def check_array_or_series_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
) -> None:
    expected_type_msg = f"a non-empty numpy.ndarray or pandas.Series with dtype '{element_type.__name__}'."
    if not isinstance(arg_values, (np.ndarray, pd.Series)):
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if isinstance(arg_values, pd.Series) and not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be {expected_type_msg}.")


def check_array_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
) -> None:
    expected_type_msg = (
        f"a non-empty numpy.ndarray with dtype '{element_type.__name__}'."
    )
    if not isinstance(arg_values, np.ndarray) or not arg_values.dtype == element_type:
        raise TypeError(f"Error: {arg_name} is expected to be {expected_type_msg}.")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be {expected_type_msg}.")


def check_array_or_dataframe_float(
    arg_name: str, arg_values: Any, element_type: Type[Any] = np.float64
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
def check_list_or_array_str(arg_name: str, arg_values: Any) -> None:
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


def check_list_or_array_int(arg_name: str, arg_values: Any) -> None:
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


def check_list_int(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, List) or not all(
        isinstance(val, int) for val in arg_values
    ):
        raise TypeError(f"Error: {arg_name} is expected to be a non-empty List[int].")
    if len(arg_values) == 0:
        raise ValueError(f"Error: {arg_name} is expected to be a non-empty List[int].")


# Datetime types:
def check_datetime_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (str, datetime.datetime)):
        raise TypeError(
            f"Error: {arg_name} is expected to be of type str or datetime.datetime."
        )


# Numberic types:
def check_float_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (float, np.floating)):
        raise TypeError(f"Error: {arg_name} is expected to be of type float.")


def check_integer_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (int, np.integer)):
        raise TypeError(f"Error: {arg_name} is expected to be of type integer.")


def check_numeric_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, (int, np.integer, float, np.floating)):
        raise TypeError(
            f"Error: {arg_name} is expected to be of type integer or float."
        )


# Boolean types:
def check_bool_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, bool):
        raise TypeError(f"Error: {arg_name} is expected to be of type bool.")


# String types:
def check_string_type(arg_name: str, arg_values: Any) -> None:
    if not isinstance(arg_values, str):
        raise TypeError(f"Error: {arg_name} is expected to be of type str.")


# Callable types:
def check_callable_type(arg_name: str, arg_values: Any) -> None:
    if not callable(arg_values):
        raise TypeError(f"Error: {arg_name} is expected to be a Callable function.")


ValidationFunc = Callable[..., None]

def type_validation(**kwargs: Any) -> None:
    # Definition of potential arguments and corresponding expected types
    type_dict: Dict[str, ValidationFunc] = {
        # DataFrames, Series, Arrays:
        "data": check_series_or_dataframe_float,
        "pf_allocation": check_dataframe_any_subtype,
        "returns_df": check_dataframe_float,
        "returns_series": check_series_float,
        "market_daily_returns": check_series_float,
        "means": check_array_or_series_float,
        "weights": check_array_or_series_float,
        "initial_weights": check_array_float,
        "weights_array": check_array_float,
        "cov_matrix": check_array_or_dataframe_float,
        # Lists:
        "names": check_list_or_array_str,
        "cols": check_list_or_array_str,
        "spans": check_list_int,
        "targets": check_list_or_array_int,
        # Datetime objects:
        "start_date": check_datetime_type,
        "end_date": check_datetime_type,
        # Strings:
        "data_api": check_string_type,
        "market_index": check_string_type,
        "method": check_string_type,
        "name": check_string_type,
        # FLOATs
        "expected_return": check_float_type,
        "volatility": check_float_type,
        "risk_free_rate": check_float_type,
        "downside_risk": check_float_type,
        "mu": check_float_type,
        "sigma": check_float_type,
        "conf_level": check_float_type,
        # INTs:
        "freq": check_integer_type,
        "span": check_integer_type,
        "num_trials": check_integer_type,
        # NUMERICs:
        "investment": check_numeric_type,
        "dividend": check_numeric_type,
        "target": check_numeric_type,
        # Booleans:
        "plot": check_bool_type,
        "save_weights": check_bool_type,
        "verbose": check_bool_type,
        "defer_update": check_bool_type,
        # Callables:
        "fun": check_callable_type,
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
