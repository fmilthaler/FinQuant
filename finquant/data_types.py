"""
This module defines type aliases and utility functions for working with arrays, data frames,
and various numeric types in Python, utilizing the 'numpy', 'numpy.typing', and 'pandas' libraries.

Type Aliases:
-------------
- ``ARRAY_OR_SERIES``: A type alias representing either a NumPy ``ndarray`` or a pandas ``Series``.
- ``ARRAY_OR_DATAFRAME``: A type alias representing either a NumPy ``ndarray`` or a pandas ``DataFrame``.
- ``ARRAY_OR_LIST``: A type alias representing either a NumPy ``ndarray`` or a ``List``.
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

>>> from finquant.data_types import ARRAY_OR_DATAFRAME, NUMERIC
# Use the defined type aliases
def process_data(data: ARRAY_OR_DATAFRAME) -> NUMERIC:
    # Process the data and return a numeric result
    return 5.0

"""
# pylint: disable=C0103


from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

# Type Aliases:
ARRAY_OR_LIST = Union[np.ndarray, List]
ARRAY_OR_SERIES = Union[np.ndarray, pd.Series]
ARRAY_OR_DATAFRAME = Union[np.ndarray, pd.DataFrame]
SERIES_OR_DATAFRAME = Union[pd.Series, pd.DataFrame]

# Numeric types
FLOAT = Union[np.floating, float]
INT = Union[np.integer, int]
NUMERIC = Union[INT, FLOAT]

# String/Datetime types
STRING_OR_DATETIME = Union[str, datetime]
