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
X = np.concatenate((Xs, Xt))
y = np.array([0.2 * x if x<0.5 else 10 for x in X.ravel()])


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, y[:100])
    assert np.abs(lr.coef_[0] - 10) < 1


def test_fit():
    np.random.seed(0)
    model = KMM(LinearRegression, fit_intercept=False)
    model.fit(X, y, range(100), range(100, 200))
    assert model.estimator_.coef_[0] - 0.2 < 1
    assert model.weights_[:50].sum() > 50
    assert model.weights_[50:].sum() < 0.1
    assert np.abs(model.predict(Xt) - y[100:]).sum() < 10