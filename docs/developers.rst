.. _developers:

####################
Notes for Developers
####################

.. note:: Contributions are welcome. If you want to add new functionality please

    1. read through `CONTRIBUTIONS.md` in the root directory of the repository, and
    2. familiarize yourself with the custom data types defined in FinQuant, and how type validation is achieved. You find relevant information below.

**********
Data Types
**********

Various custom data types are defined in ``finquant.data_types`` and used in FinQuant as type hints.

Description
###########

.. automodule:: finquant.data_types



Code Definitions
################

Array/List-Like Types
---------------------

.. autodata:: finquant.data_types.ARRAY_OR_LIST
   :annotation:

.. autodata:: finquant.data_types.ARRAY_OR_DATAFRAME
   :annotation:

.. autodata:: finquant.data_types.ARRAY_OR_SERIES
   :annotation:

.. autodata:: finquant.data_types.SERIES_OR_DATAFRAME
   :annotation:

List of Dict keys
-----------------

.. autodata:: finquant.data_types.LIST_DICT_KEYS
   :annotation:

Numeric Types
-------------

.. autodata:: finquant.data_types.FLOAT
   :annotation:

.. autodata:: finquant.data_types.INT
   :annotation:

.. autodata:: finquant.data_types.NUMERIC
   :annotation:


***************
Type validation
***************

This module provides a function ``type_validation`` that allow to effortlessly implement type validation.

Description
###########

.. automodule:: finquant.type_utilities


Code Definitions
################

.. autodata:: finquant.type_utilities.type_validation
   :annotation:
