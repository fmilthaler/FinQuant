"""

``finquant.type_utilities`` Module

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
        freq=10,
    )

"""

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
            np.issubdtype(value_type, element_type)
            for value_type in arg_values.dtypes
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
    "data": ((pd.Series, pd.DataFrame), np.number),
    "pf_allocation": (pd.DataFrame, None),
    "returns_df": (pd.DataFrame, np.floating),
    "returns_series": (pd.Series, np.floating),
    "market_daily_returns": (pd.Series, np.floating),
    "means": ((np.ndarray, pd.Series), np.floating),
    "weights": ((np.ndarray, pd.Series), np.floating),
    "initial_weights": (np.ndarray, np.floating),
    "weights_array": (np.ndarray, np.floating),
    "cov_matrix": ((np.ndarray, pd.DataFrame), np.floating),
    "df": (pd.DataFrame, None),
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
    "beta_parameter": ((float, np.floating), None),
    # INTs:
    "freq": ((int, np.integer), None),
    "span": ((int, np.integer), None),
    "num_trials": ((int, np.integer), None),
    "longer_ema_window": ((int, np.integer), None),
    "shorter_ema_window": ((int, np.integer), None),
    "signal_ema_window": ((int, np.integer), None),
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

    This function performs various type validations on a set of input variables. It helps to ensure that the input
    values conform to the expected types and conditions, raising a TypeError with a descriptive error message
    if any type validation fails and a ValueError if a numpy.array or pd.Series/DataFrame is empty.

    :param kwargs: Arbitrary keyword arguments representing the input variables to be checked.

    Raises:
        ``TypeError``:
            If any of the type validations fail, a TypeError is raised with a descriptive error message
            indicating the expected type and conditions for each variable.
        ``ValueError``:
            If any of the value validations fail, a ValueError is raised with a descriptive error message
            indicating the expected conditions for each variable.

    Example usage:

    .. code-block:: python

        type_validation(
            data=pd.DataFrame([1., 2.]),
            names=["name1", "name2"],
            start_date="2023-08-01",
            freq=10,
        )
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
