"""
Test functions for tradaboost module.
"""

import copy
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import tensorflow as tf

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
    
    
def test_tradaboost_fit_keras_model():
    np.random.seed(0)
    est = tf.keras.Sequential()
    est.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    est.compile(loss="bce", optimizer="adam")
    model = TrAdaBoost(est, n_estimators=2, random_state=0)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    yp = model.predict(Xt)
    
    est = tf.keras.Sequential()
    est.add(tf.keras.layers.Dense(2, activation="softmax"))
    est.compile(loss="mse", optimizer="adam")
    model = TrAdaBoost(est, n_estimators=2, random_state=0)
    model.fit(Xs, np.random.random((100, 2)),
              Xt=Xt[:10], yt=np.random.random((10, 2)))
    
    
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
    assert np.all(model.predict_weights(domain="src") ==
                  model.sample_weights_src_[-1])
    assert np.all(model.predict_weights(domain="tgt") ==
                  model.sample_weights_tgt_[-1])


def test_twostagetradaboostr2_fit():
    np.random.seed(0)
    model = TwoStageTrAdaBoostR2(LinearRegression(fit_intercept=False),
                         n_estimators=10)
    model.fit(Xs, ys_reg.ravel(), Xt=Xt[:10], yt=yt_reg[:10].ravel())    
    assert np.abs(model.estimators_[-1].estimators_[-1].coef_[0]
           - 1.) < 1
    assert np.abs(model.sample_weights_src_[-1][:50].sum() / 
            model.sample_weights_src_[-1][50:].sum()) > 10
    assert model.sample_weights_tgt_[-1].sum() > 0.7
    assert np.abs(model.predict(Xt) - yt_reg).sum() < 1
    argmin = np.argmin(model.estimator_errors_)
    assert np.all(model.predict_weights(domain="src") ==
                  model.sample_weights_src_[argmin])
    assert np.all(model.predict_weights(domain="tgt") ==
                  model.sample_weights_tgt_[argmin])
    
    
def test_tradaboost_deepcopy():
    np.random.seed(0)
    model = TrAdaBoost(LogisticRegression(penalty='none',
                       solver='lbfgs'),
                       n_estimators=20)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    copy_model = copy.deepcopy(model)
    assert np.all(model.predict(Xt) == copy_model.predict(Xt))
    assert hex(id(model)) != hex(id(copy_model))
