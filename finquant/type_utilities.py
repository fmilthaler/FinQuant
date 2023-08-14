import datetime
from typing import Any, Callable, Dict, List, Type, Tuple

import numpy as np
import pandas as pd


def _check_type(arg_name: str, arg_values: Any, expected_type: Type[Any], element_type: Type[Any] = None) -> None:
    if isinstance(expected_type, Tuple):
        class_names = [cls.__name__ for cls in expected_type]
        expected_type_string = ", ".join(class_names)
    else:
        expected_type_string = expected_type.__name__

    element_type_string = None
    if element_type is not None:
        if isinstance(element_type, Tuple):
            class_names = [cls.__name__ for cls in element_type]
            element_type_string = ", ".join(class_names)
        else:
            element_type_string = element_type.__name__

    validation_failed = False

    if not isinstance(arg_values, expected_type):
        validation_failed = True

    if element_type is not None:
        if isinstance(arg_values, pd.DataFrame) and not all(arg_values.dtypes == element_type):
            validation_failed = True

        if isinstance(arg_values, np.ndarray):
            if arg_values.ndim == 2 and not arg_values.dtype == element_type:
                validation_failed = True
            elif arg_values.ndim == 1 and not all(isinstance(val, element_type) for val in arg_values):
                validation_failed = True

        elif isinstance(arg_values, List) and not all(isinstance(val, element_type) for val in arg_values):
            validation_failed = True

    if validation_failed:
        error_msg = f"Error: {arg_name} is expected to be {expected_type_string}"
        if element_type_string:
            error_msg += f" with dtype '{element_type_string}'"
        raise TypeError(error_msg)


# Define a dictionary mapping each argument name to its expected type and, if applicable, element type
type_dict: Dict[str, Tuple[Type[Any], Type[Any]]] = {
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
    # Callables:
    "fun": (Callable, None),
}


def type_validation(**kwargs: Any) -> None:
    """
    Perform generic type validations on input variables.
    (same as before)
    """

    for arg_name, arg_values in kwargs.items():
        if arg_name not in type_dict:
            raise ValueError(
                f"Error: '{arg_name}' is not a valid argument. Please only use argument names defined in `type_dict`."
            )

        # Some arguments are allowed to be None, so skip them
        if arg_values is None:
            continue

        # Extract the expected type for the given argument name from type_dict
        expected_type, element_type = type_dict[arg_name]
        #expected_type = type_dict[arg_name]

        # # Check if element_type is specified in the type_dict
        # if isinstance(expected_type, Tuple) and isinstance(expected_type[0], Tuple):
        #     element_type = expected_type[1] if len(expected_type) > 1 else None
        #     expected_type = expected_type[0]
        # else:
        #     element_type = None

        # Perform the type validation using the single _check_type function
        _check_type(arg_name, arg_values, expected_type, element_type)
