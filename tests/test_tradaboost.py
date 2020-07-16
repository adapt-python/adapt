"""
Test functions for kliep module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from adapt.instance_based import (TrAdaBoost,
                                  TrAdaBoostR2,
                                  TwoStageTrAdaBoostR2)

np.random.seed(0)
Xs = np.concatenate((
    np.random.randn(50)*0.1,
    np.random.randn(50)*0.1 + 1.,
)).reshape(-1, 1)
Xt = (np.random.randn(100) * 0.1).reshape(-1, 1)
X = np.concatenate((Xs, Xt))
y_reg = np.array([0.2 * x if x<0.5 else 10 for x in X.ravel()])
y_classif = np.array(
    [x<0 if x<0.5 else x<1 for x in X.ravel()]
).astype(float)


def test_setup():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xs, y_reg[:100])
    assert np.abs(lr.coef_[0] - 10) < 1
    
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(Xs, y_classif[:100])
    assert (lr.predict(Xt) == y_classif[100:]).sum() < 70


def test_tradaboost_fit():
    np.random.seed(0)
    model = TrAdaBoost(LogisticRegression,
                       n_estimators=20,
                       penalty='none',
                       solver='lbfgs')
    model.fit(X, y_classif, range(100), range(100, 110))
    assert len(model.sample_weights_src_[0]) == 100
    assert (model.sample_weights_src_[0][:50].sum() ==
            model.sample_weights_src_[0][50:].sum())
    assert len(model.sample_weights_tgt_[-1]) == 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert (model.predict(Xt) == y_classif[100:]).sum() > 90
    
    
def test_tradaboostr2_fit():
    np.random.seed(0)
    model = TrAdaBoostR2(LinearRegression,
                         n_estimators=20,
                         fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert model.estimators_[-1].coef_[0] - 0.2 < 1
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt).ravel()
                  - y_reg[100:]).sum() < 1


def test_tradaboostr2_fit():
    np.random.seed(0)
    model = TrAdaBoostR2(LinearRegression,
                         n_estimators=20,
                         fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert model.estimators_[-1].coef_[0] - 0.2 < 1
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt).ravel()
                  - y_reg[100:]).sum() < 1


def test_twostagetradaboostr2_fit():
    np.random.seed(0)
    model = TwoStageTrAdaBoostR2(LinearRegression,
                         n_estimators=10,
                         fit_intercept=False)
    model.fit(X, y_reg, range(100), range(100, 110))
    assert (model.estimators_[-1].estimators_[-1].coef_[0]
            - 0.2 < 1)
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt).ravel()
                  - y_reg[100:]).sum() < 1
