"""
Test functions for tradaboost module.
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
ys_reg = np.array([1. * x if x<0.5 else
                   10 for x in Xs.ravel()]).reshape(-1, 1)
yt_reg = np.array([1. * x if x<0.5 else
                   10 for x in Xt.ravel()]).reshape(-1, 1)
ys_classif = np.array(
    [x<0 if x<0.5 else x<1 for x in Xs.ravel()]
).astype(float)
yt_classif = np.array(
    [x<0 if x<0.5 else x<1 for x in Xt.ravel()]
).astype(float)

def test_tradaboost_fit():
    np.random.seed(0)
    model = TrAdaBoost(LogisticRegression(penalty='none',
                       solver='lbfgs'),
                       n_estimators=20)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    assert len(model.sample_weights_src_[0]) == 100
    assert (model.sample_weights_src_[0][:50].sum() ==
            model.sample_weights_src_[0][50:].sum())
    assert len(model.sample_weights_tgt_[-1]) == 10
    assert model.sample_weights_tgt_[-1].sum() > 0.3
    assert (model.predict(Xt).ravel() == yt_classif).sum() > 90
    
    
def test_tradaboostr2_fit():
    np.random.seed(0)
    model = TrAdaBoostR2(LinearRegression(fit_intercept=False),
                         n_estimators=100,
                         Xt=Xt[:10], yt=yt_reg[:10])
    model.fit(Xs, ys_reg)
    assert np.abs(model.estimators_[-1].coef_[0] - 1.) < 1
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 1


def test_twostagetradaboostr2_fit():
    np.random.seed(0)
    model = TwoStageTrAdaBoostR2(LinearRegression(fit_intercept=False),
                         n_estimators=10)
    model.fit(Xs, ys_reg, Xt=Xt[:10], yt=yt_reg[:10])
    assert np.abs(model.estimators_[-1].estimators_[-1].coef_[0]
           - 1.) < 1
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 1
