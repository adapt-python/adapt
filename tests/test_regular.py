"""
Test functions for regular module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapt.parameter_based import (RegularTransferLR,
                                   RegularTransferLC,
                                   RegularTransferNN)

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
).astype(float) - 0.5)
yt_classif = np.sign(np.array(
    [x<0 if x<0.5 else x<1 for x in Xt.ravel()]
).astype(float) - 0.5)


def _get_network(input_shape=(1,), output_shape=(1,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    input_shape=input_shape,
                    use_bias=False))
    model.compile(loss="mse", optimizer=Adam(0.01))
    return model


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, ys_reg)
    assert np.abs(lr.coef_[0][0] - 10) < 1
    
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(Xs, ys_classif)
    assert (lr.predict(Xt) == yt_classif).sum() < 70


def test_regularlr_fit():
    np.random.seed(0)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, ys_reg)
    model = RegularTransferLR(lr, lambda_=0.)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 1
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 2
    
    model = RegularTransferLR(lr, lambda_=1000000)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0][0] - 10) < 1
    assert np.abs(model.estimator_.coef_[0][0] - lr.coef_[0][0]) < 0.001
    
    model = RegularTransferLR(lr, lambda_=1.)
    model.fit(Xt, yt_reg)
    assert np.abs(model.estimator_.coef_[0][0] - 4) < 1


def test_regularlc_fit():
    np.random.seed(0)
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(Xs, ys_classif)
    model = RegularTransferLC(lr, lambda_=0)
    model.fit(Xt, yt_classif)
    assert (model.predict(Xt).ravel() == yt_classif).sum() > 90
    
    model = RegularTransferLC(lr, lambda_=100000000)
    model.fit(Xt, yt_classif)
    assert (model.predict(Xt) == yt_classif).sum() < 70
    assert np.abs(model.estimator_.coef_[0][0] - lr.coef_[0][0]) < 0.001
    assert np.abs(model.estimator_.intercept_ - lr.intercept_[0]) < 0.001
    
    model = RegularTransferLC(lr, lambda_=1.2)
    model.fit(Xt, yt_classif)
    assert np.abs(
        (model.predict(Xt) == yt_classif).sum() - 55) < 2


def test_regularnn_fit():
    tf.random.set_seed(0)
    np.random.seed(0)
    network = _get_network()
    network.fit(Xs, ys_reg, epochs=100, batch_size=100, verbose=0)
    model = RegularTransferNN(network, lambdas=0.)
    model.fit(Xt, yt_reg, epochs=100, batch_size=100, verbose=0)
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 1
