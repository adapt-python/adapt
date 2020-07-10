"""
Test functions for utils module.
"""


import numpy as np
import pytest
from sklearn.linear_model import Ridge
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.keras.optimizers import Adam


from adapt.utils import *


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


def _get_model_Model(compiled=True):
    inputs = Input((10,))
    output = Dense(1)(inputs)
    model = Model(inputs, output)
    if compiled:
        model.compile(loss="mse", optimizer="adam")
    return model


def _get_model_Sequential():
    model = Sequential()
    model.add(Dense(1, input_shape=(10,)))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_model_with_kwargs(input_shape, output_shape,
                           name="test"):
    inputs = Input(input_shape)
    flat = Flatten()(inputs)
    output = Reshape(output_shape)(flat)
    model = Model(inputs, output, name=name)
    model.compile(loss="mse", optimizer="adam")
    return model


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
        check_estimator(_get_no_estimator)
    assert "methods" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_estimator(_DummyClassWhithoutFit)
    assert "methods" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_estimator(_DummyClassWhithoutPredict)
    assert "methods" in str(excinfo.value)


def test_check_estimator_kwargs():
    est = check_estimator(_get_estimator, alpha=123)
    assert isinstance(est, Ridge)
    assert est.alpha == 123
    est = check_estimator(Ridge, alpha=123)
    assert isinstance(est, Ridge)
    assert est.alpha == 123


@pytest.mark.parametrize("constructor",
                         [_get_model_Model, _get_model_Sequential])
def test_check_network_call(constructor):
    model = check_network(constructor)
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)
    assert model.loss == "mse"
    assert isinstance(model.optimizer, Adam)
    if len(model.layers) == 2:
        assert isinstance(model.layers[0], InputLayer)
        assert isinstance(model.layers[1], Dense)
    else:
        assert isinstance(model.layers[0], Dense)


def test_check_network_call_error():
    with pytest.raises(ValueError) as excinfo:
        check_network(123)
    assert "callable" in str(excinfo.value)
    assert "get_model" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        check_network("123",
                      constructor_name="get_123")
    assert "callable" in str(excinfo.value)
    assert "get_123" in str(excinfo.value)


def test_check_network_kwargs():
    model = check_network(_get_model_with_kwargs,
                          input_shape=(10, 2, 3),
                          output_shape=(3, 4, 5),
                          name="model")
    assert model.input_shape == (None, 10, 2, 3)
    assert model.output_shape == (None, 3, 4, 5)
    assert model.name == "model"


def test_check_network_kwargs_error_shape():
    with pytest.raises(ValueError) as excinfo:
        check_network(_get_model_Model,
                      input_shape=(10,))
    assert "input_shape" in str(excinfo.value)
    assert "get_model" in str(excinfo.value)
    assert not "Failed to build" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        check_network(_get_model_Model,
                      output_shape=(10,))
    assert "output_shape" in str(excinfo.value)
    assert "get_model" in str(excinfo.value)
    assert not "Failed to build" in str(excinfo.value)


def test_check_network_wrong_call():
    with pytest.raises(ValueError) as excinfo:
        check_network(_get_model_Model())
    assert "Failed to build" in str(excinfo.value)
    assert "get_model" in str(excinfo.value)
    assert "inputs" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        check_network(_get_model_Model,
                      wrong_arg="wrong_arg")
    assert "Failed to build" in str(excinfo.value)
    assert "get_model" in str(excinfo.value)
    assert "wrong_arg" in str(excinfo.value)


def test_check_network_Model():
    with pytest.raises(ValueError) as excinfo:
        check_network(_get_estimator)
    assert "get_model" in str(excinfo.value)
    assert "Model" in str(excinfo.value)