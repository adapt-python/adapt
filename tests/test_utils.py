"""
Test functions for utils module.
"""


import numpy as np
import pytest
from sklearn.linear_model import Ridge

from adapt.utils import check_indexes, check_estimator


def _get_estimator(**kwargs):
    return Ridge(**kwargs)


def _get_no_estimator():
    return None


class _DummyClassWhithoutFit:
    def __init__(self):
        pass

    def predict(self):
        pass


class _DummyClassWhithoutPredict:
    def __init__(self):
        pass
    
    def fit(self):
        pass


iterables = [np.array([1, 2, 3]),
             range(3),
             [1, 2, 3],
             (1, 2, 3),
             {"a": 1, "b": 2, "c": 3}]

@pytest.mark.parametrize("index", iterables)
def test_check_indexes_iterable(index):
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


def test_check_estimator_callable():
    est = check_estimator(_get_estimator)
    assert isinstance(est, Ridge)


def test_check_estimator_class():
    est = check_estimator(Ridge)
    assert isinstance(est, Ridge)


def test_check_estimator_error():
    with pytest.raises(ValueError) as excinfo:
        check_estimator(Ridge())
    assert "callable" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_estimator([1, 2, 3])
    assert "callable" in str(excinfo.value)


def test_check_estimator_no_fit_predict():
    with pytest.raises(ValueError) as excinfo:
        est = check_estimator(_get_no_estimator)
    assert "methods" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        est = check_estimator(_DummyClassWhithoutFit)
    assert "methods" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        est = check_estimator(_DummyClassWhithoutPredict)
    assert "methods" in str(excinfo.value)


def test_check_estimator_kwargs():
    est = check_estimator(_get_estimator, alpha=123)
    assert isinstance(est, Ridge)
    assert est.alpha == 123
    est = check_estimator(Ridge, alpha=123)
    assert isinstance(est, Ridge)
    assert est.alpha == 123
