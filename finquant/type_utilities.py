import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

# supress some pylint complaints for this module only
# pylint: disable=C0302,R0904,,R0912,W0212


def _check_type(
    arg_name: str,
    arg_values: Any,
    expected_type: Union[Type[Any], Tuple[Type[Any], ...]],
    element_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
) -> None:
    if isinstance(expected_type, tuple):
        class_names = [cls.__name__ for cls in expected_type]
        expected_type_string = ", ".join(class_names)
    else:
        expected_type_string = expected_type.__name__

    element_type_string = None
    if element_type is not None:
        if isinstance(element_type, tuple):
            class_names = [cls.__name__ for cls in element_type]
            element_type_string = ", ".join(class_names)
        else:
            element_type_string = element_type.__name__

    validation_failed = False

    if not isinstance(arg_values, expected_type):
        validation_failed = True

    if element_type is not None:
        if isinstance(arg_values, pd.DataFrame) and not all(
            arg_values.dtypes == element_type
        ):
            validation_failed = True

        if isinstance(arg_values, np.ndarray):
            if arg_values.ndim == 2 and not arg_values.dtype == element_type:
                validation_failed = True
            elif arg_values.ndim == 1 and not all(
                isinstance(val, element_type) for val in arg_values
            ):
                validation_failed = True

        elif isinstance(arg_values, List) and not all(
            isinstance(val, element_type) for val in arg_values
        ):
            validation_failed = True

    if validation_failed:
        error_msg = f"Error: {arg_name} is expected to be {expected_type_string}"
        if element_type_string:
            error_msg += f" with dtype '{element_type_string}'"
        raise TypeError(error_msg)


def _check_callable_type(
    arg_name: str,
    arg_values: Any,
) -> None:
    if not callable(arg_values):
        error_msg = f"Error: {arg_name} is expected to be Callable"
        raise TypeError(error_msg)


def _check_empty_data(arg_name: str, arg_values: Any) -> None:
    if isinstance(arg_values, (List, np.ndarray, pd.Series, pd.DataFrame)):
        if len(arg_values) == 0:
            raise ValueError(
                f"Error: {arg_name} is an empty list, numpy array, pandas series, or dataframe"
            )


# Define a dictionary mapping each argument name to its expected type and, if applicable, element type
type_dict: Dict[
    str,
    Tuple[
        Union[Type[Any], Tuple[Type[Any], ...]],
        Optional[Union[Type[Any], Tuple[Type[Any], ...], None]],
    ],
] = {
    # DataFrames, Series, Array:
    "data": ((pd.Series, pd.DataFrame), np.floating),
    "pf_allocation": (pd.DataFrame, None),
    "returns_df": (pd.DataFrame, np.floating),
    "returns_series": (pd.Series, np.floating),
    "market_daily_returns": (pd.Series, np.floating),
    "means": ((np.ndarray, pd.Series), np.floating),
    "weights": ((np.ndarray, pd.Series), np.floating),
    "initial_weights": (np.ndarray, np.floating),
    "weights_array": (np.ndarray, np.floating),
    "cov_matrix": ((np.ndarray, pd.DataFrame), np.floating),
    # Lists:
    "names": ((List, np.ndarray), str),
    "cols": ((List, np.ndarray), str),
    "spans": ((List, np.ndarray), (int, np.integer)),
    "targets": ((List, np.ndarray), (int, np.integer)),
    # Datetime objects:
    "start_date": ((str, datetime.datetime), None),
    "end_date": ((str, datetime.datetime), None),
    # Strings:
    "data_api": (str, None),
    "market_index": (str, None),
    "method": (str, None),
    "name": (str, None),
    # FLOATs
    "expected_return": ((float, np.floating), None),
    "volatility": ((float, np.floating), None),
    "risk_free_rate": ((float, np.floating), None),
    "downside_risk": ((float, np.floating), None),
    "mu": ((float, np.floating), None),
    "sigma": ((float, np.floating), None),
    "conf_level": ((float, np.floating), None),
    # INTs:
    "freq": ((int, np.integer), None),
    "span": ((int, np.integer), None),
    "num_trials": ((int, np.integer), None),
    # NUMERICs:
    "investment": ((int, np.integer, float, np.floating), None),
    "dividend": ((int, np.integer, float, np.floating), None),
    "target": ((int, np.integer, float, np.floating), None),
    # Booleans:
    "plot": (bool, None),
    "save_weights": (bool, None),
    "verbose": (bool, None),
    "defer_update": (bool, None),
}

type_callable_dict: Dict[
    str,
    Tuple[
        Callable[..., Any],
        Optional[Type[Any]],
    ],
] = {
    # Callables:
    "fun": (callable, None),
}


def type_validation(**kwargs: Any) -> None:
    """
    Perform generic type validations on input variables.
    (same as before)
    """

    for arg_name, arg_values in kwargs.items():
        if arg_name not in type_dict and arg_name not in type_callable_dict:
            raise ValueError(
                f"Error: '{arg_name}' is not a valid argument. "
                f"Please only use argument names defined in `type_dict` or `type_callable_dict`."
            )

        # Some arguments are allowed to be None, so skip them
        if arg_values is None:
            continue

        # Perform the type validation
        if arg_name == "fun":
            _check_callable_type(arg_name, arg_values)
        else:
            expected_type, element_type = type_dict[arg_name]
            # Validation of type
            _check_type(arg_name, arg_values, expected_type, element_type)
            # Check for empty list/array/series/dataframe
            _check_empty_data(arg_name, arg_values)
