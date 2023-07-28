"""
This module defines type aliases and utility functions for working with arrays, data frames,
and various numeric types in Python, utilizing the 'numpy', 'numpy.typing', and 'pandas' libraries.

Type Aliases:
--------------
- ``ARRAY_OR_SERIES``: A type alias representing either a NumPy ``ndarray`` or a pandas ``Series``.
- ``ARRAY_OR_DATAFRAME``: A type alias representing either a NumPy ``ndarray`` or a pandas ``DataFrame``.

Number Types:
-------------
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
# SERIES_OR_DATAFRAME = Union[pd.Series, pd.DataFrame]

# Number types
FLOAT = Union[np.floating[Any], float]
INT = Union[np.integer[Any], int]
# NPFloat = np.floating[Any]
# NPInteger = np.integer[Any]
NUMERIC = Union[INT, FLOAT]



def mytest(x: NUMERIC, y: float) -> FLOAT:
    """
    Calculate the result of a custom test function.

    This function takes two numeric inputs, `x` and `y`, and returns the result
    of a custom test operation as a floating-point number.

    Parameters:
    -----------
    x : Number
        The first numeric input. It can be either an integer or a floating-point number.

    y : float
        The second numeric input. It must be a Python float.

    Returns:
    --------
    FloatNumber
        The result of the custom test operation as a floating-point number.

    Example:
    --------
    >>> mytest(10, 2.5)
    5.0
    >>> mytest(7.5, 1.2)
    5.0
    """
    res: FLOAT = float(5)
    return res

def mytest2(x: NUMERIC, z: FLOAT, y: float = 3.14) -> FLOAT:
    """
    This is a brief description of the function.

    :param x: A numeric parameter representing ...
    :type x: :py:data:`~.finquant.type_definitions.Number`

    :param z: A number
    :type z: :py:data:`~.finquant.type_definitions.FloatNumber`

    :param y: A floating-point parameter representing ...
    :type y: float, optional (default: 3.14)

    :return: A floating-point number representing ...
    :rtype: FloatNumber

    Example:
    --------
    >>> myfunction(42, 'hello')
    3.14
    """
    res: FLOAT = float(5)
    return res