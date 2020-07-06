"""
Test functions for utils module.
"""


import numpy as np
import pytest

from adapt.utils import check_indexes, check_estimator

iterables = [np.array([1, 2, 3]),
             range(3),
             [1, 2, 3],
             (1, 2, 3),
             {"a": 1, "b": 2, "c": 3}]

@pytest.mark.parametrize("index", , iterables)
def test_check_indexes_iterable(index, no_index):
    check_indexes(index, [4, 5])
    check_indexes([4, 5], index)
    check_indexes([4, 5], [6, 7], index)


non_iterables = [True, 10, 1.3]

@pytest.mark.parametrize("index", non_iterables)
def test_check_indexes_non_iterable(index):
    with pytest.raises(ValueError) as excinfo:
        check_indexes(index, [4, 5])
    assert "src_index" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_indexes([4, 5], index)
    assert "tgt_index" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_indexes([4, 5], [6, 7], index)
    assert "tgt_index_labeled" in str(excinfo.value)


def test_check_indexes_warnings():
    with pytest.warns(None) as record:
        check_indexes([1, 2, 3], [4, 5, 6], [4, 5, 6])
    assert len(record) == 0
    with pytest.warns(UserWarning):
        check_indexes([1, 2, 3], [3, 5, 6])
    with pytest.warns(UserWarning):
        check_indexes([1, 2, 3], [4, 5, 6], [3, 7, 8])
