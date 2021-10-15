"""
Test functions for kliep module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

from adapt.instance_based import KLIEP


class DummyEstimator(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.y = y
        return self


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
    model = KLIEP(LinearRegression(fit_intercept=False),
                  sigmas=[10, 100])
    model.fit(Xs, ys, Xt)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
    assert np.abs(model.predict(Xt) - yt).sum() < 20
    

def test_fit_estimator_bootstrap_index():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  sigmas=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.random.random(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
    
    
def test_fit_estimator_sample_weight_zeros():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  sigmas=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.zeros(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
