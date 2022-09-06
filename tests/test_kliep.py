"""
Test functions for kliep module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

from adapt.instance_based import KLIEP

import pytest
import warnings

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
                  Xt,
                  sigmas=[10, 100])
    model.fit(Xs, ys)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
    assert np.abs(model.predict(Xt) - yt).sum() < 20
    assert np.all(model.weights_ == model.predict_weights())
    assert np.all(model.weights_ == model.predict_weights(Xs))
    
    
def test_fit_OG():
    np.random.seed(0)
    model = KLIEP(LinearRegression(fit_intercept=False),
                  Xt,
                  sigmas=[10, 100],
                  algo="original")
    model.fit(Xs, ys)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
    assert np.abs(model.predict(Xt) - yt).sum() < 20
    assert np.all(model.weights_ == model.predict_weights())
    assert np.all(model.weights_ == model.predict_weights(Xs))
    

def test_fit_PG():
    np.random.seed(0)
    model = KLIEP(LinearRegression(fit_intercept=False),
                  Xt,
                  sigmas=[10, 100],
                  algo="PG")
    model.fit(Xs, ys)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
    assert np.abs(model.predict(Xt) - yt).sum() < 20
    assert np.all(model.weights_ == model.predict_weights())
    assert np.all(model.weights_ == model.predict_weights(Xs))
    
    
def test_centers():
    np.random.seed(0)
    
    with pytest.raises(ValueError) as excinfo:
        model = KLIEP(gamma=10**16)
        model.fit_weights(Xs, Xt)
        assert ("No centers found! Please change the value of kernel parameter." in str(excinfo.value))
    
    with warnings.catch_warnings(record=True) as w:
        model = KLIEP(gamma=10**6)
        model.fit_weights(Xs, Xt)
        assert ("Not enough centers" in str(w[-1].message))
    

def test_fit_estimator_bootstrap_index():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  Xt,
                  sigmas=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.random.random(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
    
    
def test_fit_estimator_sample_weight_zeros():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  Xt,
                  sigmas=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.zeros(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
    
    
def test_fit_gamma():
    np.random.seed(0)
    model = KLIEP(LinearRegression(fit_intercept=False),
                  Xt,
                  gamma=[10, 100])
    model.fit(Xs, ys)
    assert np.abs(model.estimator_.coef_[0][0] - 0.2) < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
    assert np.abs(model.predict(Xt) - yt).sum() < 20
    assert np.all(model.weights_ == model.predict_weights())
    assert np.all(model.weights_ == model.predict_weights(Xs))
    
    
def test_fit_estimator_bootstrap_index_gamma():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  Xt,
                  gamma=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.random.random(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
    
    
def test_fit_estimator_sample_weight_zeros_gamma():
    np.random.seed(0)
    ys_ = np.random.randn(100)
    model = KLIEP(DummyEstimator(),
                  Xt,
                  gamma=[10, 100])
    model.fit_estimator(Xs, ys_, sample_weight=np.zeros(len(ys)))
    assert len(set(list(model.estimator_.y.ravel())) & set(list(ys_.ravel()))) > 33
    
    
def test_only_one_param():
    np.random.seed(0)
    model = KLIEP(LinearRegression(fit_intercept=False),
                  Xt,
                  kernel="poly",
                  gamma=0.1,
                  coef0=1.,
                  degree=3)
    model.fit(Xs, ys)
    assert model.best_params_ == {'coef0': 1.0, 'degree': 3, 'gamma': 0.1}
    assert model.j_scores_ == {}
    
    
def test_multiple_params():
    np.random.seed(0)
    model = KLIEP(LinearRegression(fit_intercept=False),
                  Xt,
                  kernel="sigmoid",
                  gamma=[0.1, 1.],
                  coef0=1.)
    model.fit(Xs, ys)
    assert "{'gamma': 0.1, 'coef0': 1.0}" in model.j_scores_
    assert "{'gamma': 1.0, 'coef0': 1.0}" in model.j_scores_
