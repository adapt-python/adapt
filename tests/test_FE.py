"""
Test functions for fe module.
"""


import numpy as np
import pytest
from sklearn.linear_model import Ridge, LinearRegression

from adapt.feature_based import FE


X = np.ones((100, 10))
y = np.concatenate((np.ones(75), np.zeros(25)))
  

def test_fit():   
    model = FE(Ridge, fit_intercept=False)
    model.fit(X, y, range(75), range(75, 100))
    assert isinstance(model.estimator_, Ridge)
    assert len(model.estimator_.coef_) == 3 * X.shape[1]
    assert np.abs(np.sum(model.estimator_.coef_[:X.shape[1]]) + 
                  np.sum(model.estimator_.coef_[-X.shape[1]:]) - 1) < 0.01
    assert np.abs(np.sum( model.estimator_.coef_[X.shape[1]:])) < 0.01
    
    
def test_fit_index():   
    model = FE(Ridge, fit_intercept=False)
    model.fit(X, y, range(75, 100), range(75))
    assert isinstance(model.estimator_, Ridge)
    assert len(model.estimator_.coef_) == 3 * X.shape[1]
    assert np.abs(np.sum(model.estimator_.coef_[:X.shape[1]]) +
                  np.sum(model.estimator_.coef_[-X.shape[1]:])) < 0.01
    assert np.abs(np.sum( model.estimator_.coef_[X.shape[1]:]) - 1) < 0.01


def test_fit_default():
    model = FE()
    model.fit(X, y, range(75), range(75, 100))
    assert isinstance(model.estimator_, LinearRegression)


def test_fit_sample_weight():
    model = FE(Ridge)
    model.fit(X, y, range(75), range(75, 100),
              sample_weight=np.array([0] * 75 + [1e4] * 25))
    assert np.all(model.estimator_.coef_ == 0)


def test_fit_params():
    model = FE(Ridge, alpha=123)
    model.fit(X, y, range(75), range(75, 100))
    assert model.estimator_.alpha == 123


def test_predict():
    model = FE(Ridge)
    model.fit(X, y, range(75), range(75, 100))
    y_pred = model.predict(X)
    assert np.all(y_pred < 0.01)
    y_pred = model.predict(X, domain="target")
    assert np.all(y_pred < 0.01)
    y_pred = model.predict(X, domain="source")
    assert np.all(y_pred - 1 < 0.01)
    with pytest.raises(ValueError):
        y_pred = model.predict(X, domain="tirelipimpon")
