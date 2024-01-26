"""
Test functions for regular module.
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.base import clone
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

from adapt.utils import make_classification_da, make_regression_da
from adapt.parameter_based import (RegularTransferLR,
                                   RegularTransferLC,
                                   RegularTransferNN,
                                   RegularTransferGP)

np.random.seed(0)
Xs = np.concatenate((
    np.random.randn(50)*0.1,
    np.random.randn(50)*0.1 + 1.,
)).reshape(-1, 1)
Xt = (np.random.randn(100) * 0.1).reshape(-1, 1)
ys_reg = np.array([0.2 * x if x<0.5 else
                   10 for x in Xs.ravel()]).reshape(-1, 1)
yt_reg = np.array([0.2 * x if x<0.5 else
                   10 for x in Xt.ravel()]).reshape(-1, 1)
ys_classif = np.sign(np.array(
    [x<0 if x<0.5 else x<1 for x in Xs.ravel()]
).astype(float) - 0.5).reshape(-1, 1)
yt_classif = np.sign(np.array(
    [x<0 if x<0.5 else x<1 for x in Xt.ravel()]
).astype(float) - 0.5).reshape(-1, 1)


def _get_network(input_shape=(1,), output_shape=(1,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    input_shape=input_shape,
                    kernel_initializer=GlorotUniform(seed=0),
                    use_bias=False))
    model.compile(loss="mse", optimizer=Adam(0.1))
    return model


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, ys_reg)
    assert np.abs(lr.coef_[0][0] - 10) < 1
    
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(Xs, ys_classif)
    assert (lr.predict(Xt) == yt_classif.ravel()).sum() < 70


def test_regularlr_fit():
    np.random.seed(0)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, ys_reg)
    model = RegularTransferLR(lr, lambda_=0.)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0] - 0.2) < 1
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 2
    
    model = RegularTransferLR(lr, lambda_=1000000)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0] - 10) < 1
    assert np.abs(model.estimator_.coef_[0] - lr.coef_[0]) < 0.001
    
    model = RegularTransferLR(lr, lambda_=1.)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0] - 4) < 1
    
    
def test_regularlr_multioutput():
    np.random.seed(0)
    X = np.random.randn(100, 5)+2.
    y = X[:, :2]
    lr = LinearRegression()
    lr.fit(X, y)
    model = RegularTransferLR(lr, lambda_=1.)
    model.fit(X, y)
    assert np.abs(model.predict(X) - y).sum() < 2
    assert np.all(model.coef_.shape == (2, 5))
    assert np.all(model.intercept_.shape == (2,))
    assert model.score(X, y) > 0.9
    
    
def test_regularlr_error():
    np.random.seed(0)
    Xs = np.random.randn(100, 5)
    Xt = np.random.randn(100, 5)
    ys = np.random.randn(100)
    yt = np.random.randn(100)
    lr = LinearRegression()
    lr.fit(Xs, ys)
    model = RegularTransferLR(lr, lambda_=1.)
    model.fit(Xt, yt)
    
    with pytest.raises(ValueError) as excinfo:
         model.fit(np.random.randn(100, 4), yt)
    assert "expected 5, got 4" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
         model.fit(Xt, np.random.randn(100, 2))
    assert "expected 1, got 2" in str(excinfo.value)


def test_regularlc_fit():
    np.random.seed(0)
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(Xs, ys_classif)
    model = RegularTransferLC(lr, lambda_=0)
    model.fit(Xt, yt_classif)
    assert (model.predict(Xt) == yt_classif.ravel()).sum() > 90
    
    model = RegularTransferLC(lr, lambda_=100000000)
    model.fit(Xt, yt_classif)
    assert (model.predict(Xt) == yt_classif.ravel()).sum() < 70
    assert np.abs(model.estimator_.coef_[0][0] - lr.coef_[0][0]) < 0.001
    assert np.abs(model.estimator_.intercept_ - lr.intercept_[0]) < 0.001
    
    model = RegularTransferLC(lr, lambda_=1.2)
    model.fit(Xt, yt_classif)
    assert (model.predict(Xt) == yt_classif.ravel()).sum() > 95
    
    
def test_regularlc_multiclass():
    np.random.seed(0)
    X = np.random.randn(100, 5)
    y = np.zeros(len(X))
    y[X[:, :2].sum(1)<0] = 1
    y[X[:, 3:].sum(1)>0] = 2
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(X, y)
    model = RegularTransferLC(lr, lambda_=1.)
    model.fit(X, y)
    assert (model.predict(X) == y).sum() > 90
    assert np.all(model.coef_.shape == (3, 5))
    assert np.all(model.intercept_.shape == (3,))
    assert model.score(X, y) > 0.9


def test_regularnn_fit():
    tf.random.set_seed(0)
    np.random.seed(0)
    network = _get_network()
    network.fit(Xs, ys_reg, epochs=100, batch_size=100, verbose=0)
    model = RegularTransferNN(network, lambdas=0., optimizer=Adam(0.1))
    model.fit(Xt, yt_reg, epochs=100, batch_size=100, verbose=0)
    # assert np.abs(network.predict(Xs) - ys_reg).sum() < 1
    assert np.sum(np.abs(network.get_weights()[0] - model.get_weights()[0])) > 4.
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 10
    
    model = RegularTransferNN(network, lambdas=10000000., optimizer=Adam(0.1))
    model.fit(Xt, yt_reg, epochs=100, batch_size=100, verbose=0)
    
    assert np.sum(np.abs(network.get_weights()[0] - model.get_weights()[0])) < 0.001
    assert np.abs(model.predict(Xt) - yt_reg).sum() > 10
    
    
def test_regularnn_reg():
    tf.random.set_seed(0)
    np.random.seed(0)
    network = _get_network()
    network.fit(Xs, ys_reg, epochs=100, batch_size=100, verbose=0)
    model = RegularTransferNN(network, regularizer="l1")
    model.fit(Xt, yt_reg, epochs=100, batch_size=100, verbose=0)
    
    with pytest.raises(ValueError) as excinfo:
         model = RegularTransferNN(network, regularizer="l3")
    assert "l1' or 'l2', got, l3" in str(excinfo.value)
    
    
def test_clone():
    Xs = np.random.randn(100, 5)
    ys = np.random.choice(2, 100)
    lr = LinearRegression()
    lr.fit(Xs, ys)
    model = RegularTransferLR(lr, lambda_=1.)
    model.fit(Xs, ys)
    
    new_model = clone(model)
    new_model.fit(Xs, ys)
    new_model.predict(Xs);
    assert model is not new_model
    
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    lr.fit(Xs, ys)
    model = RegularTransferLC(lr, lambda_=1.)
    model.fit(Xs, ys)
    
    new_model = clone(model)
    new_model.fit(Xs, ys)
    new_model.predict(Xs);
    assert model is not new_model
    
    
def test_regulargp_reg():
    Xs, ys, Xt, yt = make_regression_da()
    kernel = Matern() + WhiteKernel()
    src_model = GaussianProcessRegressor(kernel)
    src_model.fit(Xs, ys)
    score1 = src_model.score(Xt, yt)
    tgt_model = RegularTransferGP(src_model, lambda_=1.)
    tgt_model.fit(Xt[:3], yt[:3])
    score2 = tgt_model.score(Xt, yt)
    assert score1 < score2
    

def test_regulargp_classif():
    Xs, ys, Xt, yt = make_classification_da()
    kernel = Matern() + WhiteKernel()
    src_model = GaussianProcessClassifier(kernel)
    src_model.fit(Xs, ys)
    score1 = src_model.score(Xt, yt)
    tgt_model = RegularTransferGP(src_model, lambda_=1.)
    tgt_model.fit(Xt[:3], yt[:3])
    score2 = tgt_model.score(Xt, yt)
    assert score1 < score2
    

def test_regulargp_multi_classif():
    Xs, ys, Xt, yt = make_classification_da()
    ys[:5] = 3
    kernel = Matern() + WhiteKernel()
    src_model = GaussianProcessClassifier(kernel)
    src_model.fit(Xs, ys)
    score1 = src_model.score(Xt, yt)
    tgt_model = RegularTransferGP(src_model, lambda_=1.)
    tgt_model.fit(Xt[:3], yt[:3])
    score2 = tgt_model.score(Xt, yt)
    assert score1 < score2