"""
This module defines type aliases and utility functions for working with arrays, data frames,
and various numeric types in Python, utilizing the 'numpy', 'numpy.typing', and 'pandas' libraries.

Type Aliases:
-------------
- ``ARRAY_OR_SERIES``: A type alias representing either a NumPy ``ndarray`` or a pandas ``Series``.
- ``ARRAY_OR_DATAFRAME``: A type alias representing either a NumPy ``ndarray`` or a pandas ``DataFrame``.
- ``SERIES_OR_DATAFRAME``: A type alias representing either a pandas ``Series`` or a pandas ``DataFrame``.

Numeric Types:
--------------
- ``FLOAT``: A type alias representing either a NumPy floating-point number or a Python float.
- ``INT``: A type alias representing either a NumPy integer or a Python int.
- ``NUMERIC``: A type alias representing either an ``INT`` or a ``FLOAT``.

Dependencies:
-------------
This module requires the following external libraries:

- ``numpy`` (imported as ``np``)
- ``numpy.typing`` (imported as ``npt``)
- ``pandas`` (imported as ``pd``)

Usage Example:
--------------

>>> from finquant.type_definitions import ARRAY_OR_DATAFRAME, NUMERIC
# Use the defined type aliases
def process_data(data: ARRAY_OR_DATAFRAME) -> NUMERIC:
    # Process the data and return a numeric result
    return 5.0

"""


from typing import Any, List, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# Type Aliases:
ARRAY_OR_SERIES = Union[np.ndarray, pd.Series]
ARRAY_OR_DATAFRAME = Union[np.ndarray, pd.DataFrame]
ARRAY_OR_LIST = Union[np.ndarray, List]
SERIES_OR_DATAFRAME = Union[pd.Series, pd.DataFrame]

# Numeric types
FLOAT = Union[np.floating[Any], float]
INT = Union[np.integer[Any], int]
# NPFloat = np.floating[Any]
# NPInteger = np.integer[Any]
NUMERIC = Union[INT, FLOAT]
