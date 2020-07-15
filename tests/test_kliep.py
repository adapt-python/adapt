"""
Test functions for kliep module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from adapt.instance_based import KLIEP

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
    model = KLIEP(LinearRegression, sigmas=[10, 100],
                  fit_intercept=False)
    model.fit(X, y, range(100), range(100, 200))
    assert model.estimator_.coef_[0] - 0.2 < 10
    assert model.weights_[:50].sum() > 90
    assert model.weights_[50:].sum() < 0.5
