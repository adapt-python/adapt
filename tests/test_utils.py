"""
Test functions for utils module.
"""


import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree._tree import Tree
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.python.keras.engine.input_layer import InputLayer

from adapt.utils import *


def is_equal_estimator(v1, v2):
    assert type(v2) == type(v1)
    if isinstance(v1, np.ndarray):
        assert np.array_equal(v1, v2)
    elif isinstance(v1, (BaseEstimator, KerasClassifier, KerasRegressor)):
        assert is_equal_estimator(v1.__dict__, v2.__dict__)
    elif isinstance(v1, Model):
        assert is_equal_estimator(v1.get_config(),
                                  v2.get_config())
    elif isinstance(v1, dict):
        diff_keys = ((set(v1.keys())-set(v2.keys())) |
                    (set(v2.keys())-set(v1.keys())))
        for k in diff_keys:
            assert "input_shape" in k
        for k1_i, v1_i in v1.items():
            # Avoid exception due to new input layer name
            if k1_i != "name" and not "input_shape" in str(k1_i):
                v2_i = v2[k1_i]
                assert is_equal_estimator(v1_i, v2_i)
    elif isinstance(v1, (list, tuple)):
        assert len(v1) == len(v2)
        for v1_i, v2_i in zip(v1, v2):
            assert is_equal_estimator(v1_i, v2_i)
    elif isinstance(v1, Tree):
        pass # TODO create a function to check if two tree are equal
    else:
        assert v1 == v2
    return True

    

class CustomEstimator(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass

    
class DummyModel(Model):
    
    def __init__(self):
        pass
    

class CantBeDeepCopied(BaseEstimator):
    
    def __init__(self):
        pass
    
    def __deepcopy__(self):
        raise ValueError("Can not be deep copied!")


def _get_model_Model(compiled=True, custom_loss=False):
    inputs = Input((10,))
    output = Dense(1)(inputs)
    model = Model(inputs, output)
    if custom_loss:
        loss = K.mean(output)
        model.add_loss(loss)
    if compiled:
        model.compile(loss="mse", optimizer="adam")
    return model


def _get_model_Sequential(input_shape=None, compiled=True):
    model = Sequential()
    if input_shape is not None:
        model.add(Dense(1, input_shape=input_shape))
    else:
        model.add(Dense(1))
    if compiled:
        model.compile(loss="mse", optimizer="adam")
    return model


arrays_nd = [np.ones((10, 1)), np.zeros((10, 10)),
            np.zeros((10, 5, 1)), np.full((10, 20), -5.5),
            np.ones((1, 1)), np.random.randn(1, 5, 5, 1)]

@pytest.mark.parametrize("z", arrays_nd)
def test_check_arrays_nd(z):
    Xs, ys, Xt, yt = check_arrays(z, z, z, z)
    assert np.array_equal(Xs, z)
    assert np.array_equal(ys, z)
    assert np.array_equal(Xt, z)
    assert np.array_equal(yt, z)
    
    
def test_check_arrays_diff_input():
    Xs, ys, Xt, yt = arrays_nd[:4]
    assert np.array_equal(Xs, arrays_nd[0])
    assert np.array_equal(ys, arrays_nd[1])
    assert np.array_equal(Xt, arrays_nd[2])
    assert np.array_equal(yt, arrays_nd[3])
    
    
arrays_1d = [np.ones((10,)), np.ones((1,))]
arrays_2d = [np.ones((10, 1)), np.ones((1, 1))]

@pytest.mark.parametrize("z, zz", zip(arrays_1d, arrays_2d))
def test_check_arrays_1d(z, zz):
    Xs, ys, Xt, yt = check_arrays(z, z, z, z)
    assert np.array_equal(Xs, zz)
    assert np.array_equal(ys, zz)
    assert np.array_equal(Xt, zz)
    assert np.array_equal(yt, zz)


def test_check_arrays_no_yt():
    z = arrays_nd[0]
    Xs, ys, Xt, yt = check_arrays(z, z, z)
    assert yt is None
    assert np.array_equal(Xs, z)
    assert np.array_equal(ys, z)
    assert np.array_equal(Xt, z)


def test_check_arrays_length_error():
    z = arrays_nd[0]
    with pytest.raises(ValueError) as excinfo:
         Xs, ys, Xt, yt = check_arrays(z, z[:5], z)
    assert "Length of Xs and ys mismatch: 10 != 5" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
         Xs, ys, Xt, yt = check_arrays(z, z, z, z[:5])
    assert "Length of Xt and yt mismatch: 10 != 5" in str(excinfo.value)
    
    
def test_check_arrays_no_array():
    z = np.array([1,2,3])
    with pytest.raises(TypeError) as excinfo:
         Xs, ys, Xt, yt = check_arrays("123", z, z)
            

@pytest.mark.parametrize("X", arrays_nd)
def test_check_one_array_nd(X):
    Xt = check_one_array(X)
    assert np.array_equal(Xt, X)
    
    
@pytest.mark.parametrize("X, Xtt", zip(arrays_1d, arrays_2d))
def test_check_one_array_1d(X, Xtt):
    Xt = check_one_array(X)
    assert np.array_equal(Xt, Xtt)

    
networks = [
    _get_model_Model(compiled=True, custom_loss=False),
    _get_model_Sequential(compiled=True, input_shape=(10,)),
    _get_model_Sequential(compiled=True, input_shape=None),
    _get_model_Model(compiled=False, custom_loss=False),
    _get_model_Model(compiled=False, custom_loss=True),
    _get_model_Sequential(compiled=False, input_shape=(10,)),
    _get_model_Sequential(compiled=False, input_shape=None)
]
    

@pytest.mark.parametrize("net", networks)
def test_check_network_network(net):
    new_net = check_network(net, compile_=False)
    assert is_equal_estimator(new_net, net)
    if net.built:
        for i in range(len(net.get_weights())):
            assert np.array_equal(net.get_weights()[i],
                              new_net.get_weights()[i])
    net.predict(np.ones((10, 10)))
    new_net = check_network(net, compile_=False)
    assert is_equal_estimator(new_net, net)
    for i in range(len(net.get_weights())):
        assert np.array_equal(net.get_weights()[i],
                              new_net.get_weights()[i])


@pytest.mark.parametrize("net", networks)
def test_check_network_copy(net):
    new_net = check_network(net, copy=True, compile_=False)
    assert hex(id(new_net)) != hex(id(net))
    new_net = check_network(net, copy=False, compile_=False)
    assert hex(id(new_net)) == hex(id(net))
    

no_networks = ["lala", Ridge(), 123, np.ones((10, 10))]

@pytest.mark.parametrize("no_net", no_networks)
def test_check_network_no_model(no_net):
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(no_net)
    assert ("Expected `network` argument "
            "to be a `Model` instance,"
            " got: %s"%str(no_net) in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(no_net, display_name="tireli")
    assert ("Expected `tireli` argument "
            "to be a `Model` instance,"
            " got: %s"%str(no_net) in str(excinfo.value))
    

def test_check_network_force_copy():
    model = DummyModel()
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(model, copy=True, force_copy=True)
    assert ("`network` argument can't be duplicated. "
            "Recorded exception: " in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(model, copy=True, force_copy=True,
                                display_name="tireli")
    assert ("`tireli` argument can't be duplicated. "
            "Recorded exception: " in str(excinfo.value))
    
    with pytest.warns(UserWarning) as record:
        new_net = check_network(model, copy=True, force_copy=False,
                                compile_=False)
    assert ("`network` argument can't be duplicated. "
            "Recorded exception: " in str(record[0].message))
    with pytest.warns(UserWarning) as record:
        new_net = check_network(model, copy=True, force_copy=False,
                                compile_=False,
                                display_name="tireli")
    assert ("`tireli` argument can't be duplicated. "
            "Recorded exception: " in str(record[0].message))
    
    new_net = check_network(model, copy=False, force_copy=True)
    
    
def test_check_network_compile():
    net = _get_model_Sequential(compiled=False)
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(net, copy=True, compile_=True)
    assert ("The given `network` argument is not compiled yet. "
            "Please use `model.compile(optimizer, loss)`." 
            in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_net = check_network(net, copy=True, compile_=True,
                                display_name="tireli")
    assert ("The given `tireli` argument is not compiled yet. "
            "Please use `model.compile(optimizer, loss)`." 
            in str(excinfo.value))
    

estimators = [
    Ridge(),
    Ridge(alpha=10, fit_intercept=False, tol=0.1),
    DecisionTreeClassifier(max_depth=10),
    AdaBoostRegressor(Ridge(alpha=0.01)),
    TransformedTargetRegressor(regressor=Ridge(alpha=25), transformer=StandardScaler()),
    MultiOutputRegressor(Ridge(alpha=0.3)),
    make_pipeline(StandardScaler(), Ridge(alpha=0.2)),
    KerasClassifier(_get_model_Sequential, input_shape=(1,)),
    CustomEstimator()
]

@pytest.mark.parametrize("est", estimators)
def test_check_estimator_estimators(est):
    new_est = check_estimator(est, copy=True, force_copy=True)
    assert is_equal_estimator(est, new_est)
    if isinstance(est, MultiOutputRegressor):
        est.fit(np.linspace(0, 1, 10).reshape(-1, 1),
        np.stack([np.linspace(0, 1, 10)<0.5]*2, -1).astype(float))
    else:
        est.fit(np.linspace(0, 1, 10).reshape(-1, 1),
                (np.linspace(0, 1, 10)<0.5).astype(float))
    if isinstance(est, KerasClassifier):
        new_est = check_estimator(est, copy=False)
    else:
        new_est = check_estimator(est, copy=True, force_copy=True)
    assert is_equal_estimator(est, new_est)
    
    
@pytest.mark.parametrize("est", networks[:3])
def test_check_estimator_networks(est):
    new_est = check_estimator(est)
    assert is_equal_estimator(est, new_est)
    
    
no_estimators = ["lala", 123, np.ones((10, 10))]

@pytest.mark.parametrize("no_est", no_estimators)
def test_check_estimator_no_estimators(no_est):
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(no_est)
    assert ("`estimator` argument is neither a sklearn `BaseEstimator` "
            "instance nor a tensorflow Model instance. "
            "Given argument, %s"%str(no_est) in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(no_est, display_name="tireli")
    assert ("`tireli` argument is neither a sklearn `BaseEstimator` "
            "instance nor a tensorflow Model instance. "
            "Given argument, %s"%str(no_est) in str(excinfo.value))
    

@pytest.mark.parametrize("est", estimators)
def test_check_estimator_copy(est):
    new_est = check_estimator(est, copy=True)
    assert hex(id(new_est)) != hex(id(est))
    new_est = check_estimator(est, copy=False)
    assert hex(id(new_est)) == hex(id(est))
    
    
def test_check_estimator_force_copy():
    est = CantBeDeepCopied()
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(est, copy=True, force_copy=True)
    assert ("`estimator` argument can't be duplicated. "
            "Recorded exception: " in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(est, copy=True, force_copy=True,
                                  display_name="tireli")
    assert ("`tireli` argument can't be duplicated. "
            "Recorded exception: " in str(excinfo.value))
    
    with pytest.warns(UserWarning) as record:
        new_est = check_estimator(est, copy=True, force_copy=False)
    assert ("`estimator` argument can't be duplicated. "
            "Recorded exception: " in str(record[0].message))
    with pytest.warns(UserWarning) as record:
        new_est = check_estimator(est, copy=True, force_copy=False,
                                  display_name="tireli")
    assert ("`tireli` argument can't be duplicated. "
            "Recorded exception: " in str(record[0].message))
    
    new_est = check_estimator(est, copy=False, force_copy=True)
    
    
def test_check_estimator_task():
    new_est = check_estimator()
    assert isinstance(new_est, LinearRegression)
    new_est = check_estimator(task="class")
    assert isinstance(new_est, LogisticRegression)
    new_est = check_estimator(DecisionTreeClassifier(),
                              task="class")
    assert isinstance(new_est, DecisionTreeClassifier)
    new_est = check_estimator(Ridge(),
                              task="reg")
    assert isinstance(new_est, Ridge)
    
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(DecisionTreeClassifier(), task="reg")
    assert ("`estimator` argument is a sklearn `ClassifierMixin` instance "
            "whereas the considered object handles only regression task. "
            "Please provide a sklearn `RegressionMixin` instance or a "
            "tensorflow Model instance." in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(DecisionTreeClassifier(), task="reg",
                                  display_name="tireli")
    assert ("`tireli` argument is a sklearn"
            " `ClassifierMixin` instance " in str(excinfo.value))
    
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(Ridge(), task="class")
    assert ("`estimator` argument is a sklearn `RegressionMixin` instance "
            "whereas the considered object handles only classification task. "
            "Please provide a sklearn `ClassifierMixin` instance or a "
            "tensorflow Model instance." in str(excinfo.value))
    with pytest.raises(ValueError) as excinfo:
        new_est = check_estimator(Ridge(), task="class",
                                  display_name="tireli")
    assert ("`tireli` argument is a sklearn"
            " `RegressionMixin` instance " in str(excinfo.value))


def test_get_default_encoder():
    model = get_default_encoder()
    assert isinstance(model.layers[0], Flatten)
    assert isinstance(model.layers[1], Dense)
    assert model.layers[1].get_config()["units"] == 10
    assert model.layers[1].get_config()["activation"] == "relu"
    
    
def test_get_default_task():
    model = get_default_task()
    assert isinstance(model.layers[0], Flatten)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Dense)
    assert isinstance(model.layers[3], Dense)
    assert model.layers[1].get_config()["units"] == 10
    assert model.layers[1].get_config()["activation"] == "relu"
    assert model.layers[2].get_config()["units"] == 10
    assert model.layers[2].get_config()["activation"] == "relu"
    assert model.layers[3].get_config()["units"] == 1
    assert model.layers[3].get_config()["activation"] == "linear"
    
    
def test_get_default_discriminator():
    model = get_default_discriminator()
    assert isinstance(model.layers[0], Flatten)
    assert isinstance(model.layers[1], Dense)
    assert isinstance(model.layers[2], Dense)
    assert isinstance(model.layers[3], Dense)
    assert model.layers[1].get_config()["units"] == 10
    assert model.layers[1].get_config()["activation"] == "relu"
    assert model.layers[2].get_config()["units"] == 10
    assert model.layers[2].get_config()["activation"] == "relu"
    assert model.layers[3].get_config()["units"] == 1
    assert model.layers[3].get_config()["activation"] == "sigmoid"


scales = [-1, 0, 1., 0.1]

@pytest.mark.parametrize("lambda_", scales)
def test_gradienthandler(lambda_):
    grad_handler = GradientHandler(lambda_)
    inputs = K.variable([1, 2, 3])
    assert np.all(grad_handler(inputs) == inputs)
    with tf.GradientTape() as tape:
        gradient = tape.gradient(grad_handler(inputs),
                                 inputs)
    assert np.all(gradient == lambda_ * np.ones(3))


def test_make_classification_da():
    Xs, ys, Xt, yt = make_classification_da()
    assert Xs.shape == (100, 2)
    assert len(ys) == 100
    assert Xt.shape == (100, 2)
    assert len(yt) == 100
    Xs, ys, Xt, yt = make_classification_da(1000, 10)
    assert Xs.shape == (1000, 10)
    assert len(ys) == 1000
    assert Xt.shape == (1000, 10)
    assert len(yt) == 1000


def test_make_regression_da():
    Xs, ys, Xt, yt = make_regression_da()
    assert Xs.shape == (100, 1)
    assert len(ys) == 100
    assert Xt.shape == (100, 1)
    assert len(yt) == 100
    Xs, ys, Xt, yt = make_regression_da(1000, 10)
    assert Xs.shape == (1000, 10)
    assert len(ys) == 1000
    assert Xt.shape == (1000, 10)
    assert len(yt) == 1000
