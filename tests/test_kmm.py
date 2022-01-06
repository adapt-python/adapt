"""
Test functions for kmm module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from adapt.instance_based import KMM

np.random.seed(0)
Xs = np.concatenate((
    np.random.randn(50)*0.1,
    np.random.randn(50)*0.1 + 1.,
)).reshape(-1, 1)
Xt = (np.random.randn(100) * 0.1).reshape(-1, 1)
ys = np.array([0.2 * x if x<0.5
               else 10 for x in Xs.ravel()]).reshape(-1, 1)
yt = np.array([0.2 * x if x<0.5
               else 10 for x in Xt.ravel()]).reshape(-1, 1)


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, ys)
    assert np.abs(lr.coef_[0][0] - 10) < 1


def test_fit():
    np.random.seed(0)
    model = KMM(LinearRegression(fit_intercept=False), gamma=1.)
    model.fit(Xs, ys, Xt=Xt)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 1
    assert model.weights_[:50].sum() > 50
    assert model.weights_[50:].sum() < 0.1
    assert np.abs(model.predict(Xt) - yt).sum() < 10
    assert np.all(model.weights_ == model.predict_weights())
    
    
def test_tol():
    np.random.seed(0)
    model = KMM(LinearRegression(fit_intercept=False), gamma=1., tol=0.1)
    model.fit(Xs, ys, Xt=Xt)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) > 5
    
    
def test_batch():
    np.random.seed(0)
    model = KMM(LinearRegression(fit_intercept=False), gamma=1.,
                max_size=35)
    model.fit(Xs, ys, Xt=Xt)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 1
    assert model.weights_[:50].sum() > 50
    assert model.weights_[50:].sum() < 0.1
    assert np.abs(model.predict(Xt) - yt).sum() < 10
    assert np.all(model.weights_ == model.predict_weights())