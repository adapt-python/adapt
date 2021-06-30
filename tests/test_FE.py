"""
Test functions for fe module.
"""


import numpy as np
import pytest
from sklearn.linear_model import Ridge, LinearRegression

from adapt.feature_based import FE


Xs = np.ones((75, 10))
Xt = np.ones((25, 10))
ys = np.ones(75)
yt = np.zeros(25)


def test_fit():
    model = FE(Ridge(fit_intercept=False))
    model.fit(Xs, ys, Xt, yt)
    assert isinstance(model.estimator_, Ridge)
    assert len(model.estimator_.coef_[0]) == 30
    assert np.abs(model.estimator_.coef_[0][20:].sum() +
                  model.estimator_.coef_[0][:10].sum() - 1) < 0.01
    assert np.abs(model.estimator_.coef_[0][20:].sum() +
                  model.estimator_.coef_[0][10:20].sum()) < 0.01
    
    
def test_fit_default():
    model = FE()
    model.fit(Xs, ys, Xt, yt)
    assert isinstance(model.estimator_, LinearRegression)


def test_fit_params():
    model = FE(Ridge(alpha=123))
    model.fit(Xs, ys, Xt, yt)
    assert model.estimator_.alpha == 123


def test_predict():
    model = FE(Ridge())
    model.fit(Xs, ys, Xt, yt)
    y_pred = model.predict(Xt)
    assert np.all(y_pred < 0.01)
    y_pred = model.predict(Xt, domain="target")
    assert np.all(y_pred < 0.01)
    y_pred = model.predict(Xs, domain="source")
    assert np.all(y_pred - 1 < 0.01)
    with pytest.raises(ValueError):
        model.predict(Xs, domain="tirelipimpon")
