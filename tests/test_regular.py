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
X = np.concatenate((Xs, Xt))
y_reg = np.array([0.2 * x if x<0.5 else 10 for x in X.ravel()])
y_classif = np.sign(np.array(
    [x<0 if x<0.5 else x<1 for x in X.ravel()]
).astype(float) - 0.5)


def _get_network(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    input_shape=input_shape,
                    use_bias=False))
    model.compile(loss="mse", optimizer=Adam(0.01))
    return model


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, y_reg[:100])
    assert np.abs(lr.coef_[0] - 10) < 1
    
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(Xs, y_classif[:100])
    assert (lr.predict(Xt) == y_classif[100:]).sum() < 70


def test_regularlr_fit():
    np.random.seed(0)
    model = RegularTransferLR(LinearRegression,
                              intercept=False,
                              lambdap=0,
                              fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert np.abs(model.coef_[0] - 0.2) < 1
    assert np.abs(model.predict(Xt).ravel() 
                  - y_reg[100:]).sum() < 2
    
    model = RegularTransferLR(LinearRegression,
                              intercept=False,
                              lambdap=1000000,
                              fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert np.abs(model.coef_[0] - 10) < 1
    assert np.abs(model.coef_[0] -
                  model.estimator_src_.coef_[0]) < 0.001
    
    model = RegularTransferLR(LinearRegression,
                              intercept=False,
                              lambdap=1,
                              fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert np.abs(model.coef_[0] - 4) < 1


def test_regularlc_fit():
    np.random.seed(0)
    model = RegularTransferLC(LogisticRegression,
                              lambdap=0,
                              penalty='none',
                              solver='lbfgs')
    model.fit(X, y_classif, range(100), range(100, 110))
    assert (model.predict(Xt) == y_classif[100:]).sum() > 90
    
    model = RegularTransferLC(LogisticRegression,
                              lambdap=100000000,
                              penalty='none',
                              solver='lbfgs')
    model.fit(X, y_classif, range(100), range(100, 110))
    assert (model.predict(Xt) == y_classif[100:]).sum() < 70
    assert np.abs(model.coef_[0] -
                  model.estimator_src_.coef_[0][0]) < 0.001
    assert np.abs(model.intercept_ -
                  model.estimator_src_.intercept_[0]) < 0.001
    
    model = RegularTransferLC(LogisticRegression,
                              lambdap=1.2,
                              penalty='none',
                              solver='lbfgs')
    model.fit(X, y_classif, range(100), range(100, 110))
    assert np.abs(
        (model.predict(Xt) == y_classif[100:]).sum() - 80) < 2


# def test_regularnn_fit():
#     tf.random.set_seed(0)
#     np.random.seed(0)
#     model = RegularTransferNN(_get_network,
#                               lambdas=0)
#     model.fit(X, y_reg, range(100), range(100, 110),
#               epochs=100, batch_size=100, verbose=0)
#     assert np.abs(model.predict(Xt).ravel() 
#                   - y_reg[100:]).sum() < 1
