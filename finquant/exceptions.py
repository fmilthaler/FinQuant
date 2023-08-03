"""
Custom Exceptions Module

This module defines custom exception classes that represent various error scenarios
related to financial data retrieval from external APIs. The exceptions are designed
to provide specific context and information about different types of errors that
can occur during the data retrieval process.

Exceptions:
    - InvalidDateFormatError: Raised when an invalid date format is encountered during
      date parsing for financial data retrieval.
    - QuandlLimitError: Raised when the API limit for Quandl data requests is reached.
    - QuandlError: Raised for general errors that occur during Quandl data retrieval.
    - YFinanceError: Raised for general errors that occur during YFinance data retrieval.

Usage:
    These custom exceptions can be raised within the respective functions that handle
    data retrieval from external APIs, such as Quandl and YFinance. When an exception
    is raised, it provides specific information about the error, making it easier to
    diagnose and handle exceptional cases during data retrieval operations.

Example:
    try:
        # Code that may raise one of the custom exceptions.
    except InvalidDateFormatError as exc:
        # Handle the invalid date format error here.
    except QuandlLimitError as exc:
        # Handle the Quandl API limit error here.
    except QuandlError as exc:
        # Handle other Quandl-related errors here.
    except YFinanceError as exc:
        # Handle YFinance-related errors here.

"""


class InvalidDateFormatError(Exception):
    """
    Exception for Invalid Date Format

    This exception is raised when an invalid date format is encountered during date
    parsing for financial data retrieval. It is typically raised when attempting to
    convert a string to a datetime object with an incorrect format.

    Example:
        try:
            start_date = datetime.datetime.strptime("2023/08/01", "%Y-%m-%d")
        except ValueError as exc:
            raise InvalidDateFormatError("Invalid date format. Use 'YYYY-MM-DD'.") from exc
    """


class QuandlLimitError(Exception):
    """
    Exception for Quandl API Limit Reached

    This exception is raised when the API limit for Quandl data requests is reached.
    It indicates that the rate limit or request quota for the Quandl API has been
    exceeded, and no more requests can be made until the limit is reset.

    Example:
        try:
            resp = quandl.get("GOOG", start_date="2023-08-01", end_date="2023-08-05")
        except quandl.errors.QuandlLimit as exc:
            raise QuandlLimitError("Quandl API limit reached. Try again later.") from exc
    """


class QuandlError(Exception):
    """
    Exception for Quandl Data Retrieval Error

    This exception is raised for general errors that occur during Quandl data retrieval.
    It can be used to handle any unexpected issues that arise while interacting with
    the Quandl API.

    Example:
        try:
            resp = quandl.get("GOOG", start_date="2023-08-01", end_date="2023-08-05")
        except Exception as exc:
            raise QuandlError("An error occurred while retrieving data from Quandl.") from exc
    """


class YFinanceError(Exception):
    """
    Exception for YFinance Data Retrieval Error

    This exception is raised for general errors that occur during YFinance data retrieval.
    It can be used to handle any unexpected issues that arise while interacting with
    the YFinance library.

    Example:
        try:
            data = yfinance.download("GOOG", start="2023-08-01", end="2023-08-05")
        except Exception as exc:
            raise YFinanceError("An error occurred while retrieving data from YFinance.") from exc
    """
