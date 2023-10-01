from typing import List

from finquant.data_types import ELEMENT_TYPE, LIST_DICT_KEYS


def all_list_ele_in_other(
    # l_1: List, l_2: List
    l_1: LIST_DICT_KEYS[ELEMENT_TYPE],
    l_2: LIST_DICT_KEYS[ELEMENT_TYPE],
) -> bool:
    """Returns True if all elements of list l1 are found in list l2."""
    return all(ele in l_2 for ele in l_1)
