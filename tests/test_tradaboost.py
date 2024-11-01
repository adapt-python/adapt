"""
Test functions for tradaboost module.
"""

import copy
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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
    model = TrAdaBoost(LogisticRegression(penalty=None,
                       solver='lbfgs'),
                       n_estimators=20)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    score = model.score(Xs, ys_classif)
    assert score == accuracy_score(ys_classif, model.predict(Xs))
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
    est.compile(loss="bce", optimizer=Adam())
    model = TrAdaBoost(est, n_estimators=2, random_state=0)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    yp = model.predict(Xt)
    
    est = tf.keras.Sequential()
    est.add(tf.keras.layers.Dense(2, activation="softmax"))
    est.compile(loss="mse", optimizer=Adam())
    model = TrAdaBoost(est, n_estimators=2, random_state=0)
    model.fit(Xs, np.random.random((100, 2)),
              Xt=Xt[:10], yt=np.random.random((10, 2)))
    
    score = model.score(Xs, ys_classif)
    assert score == accuracy_score(ys_classif, model.predict(Xs))
    
    
def test_tradaboostr2_fit():
    np.random.seed(0)
    model = TrAdaBoostR2(LinearRegression(fit_intercept=False),
                         n_estimators=100,
                         Xt=Xt[:10], yt=yt_reg[:10])
    model.fit(Xs, ys_reg)
    score = model.score(Xs, ys_reg)
    assert score == r2_score(ys_reg, model.predict(Xs))
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
    score = model.score(Xs, ys_reg)
    assert score == r2_score(ys_reg, model.predict(Xs))
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
    model = TrAdaBoost(LogisticRegression(penalty=None,
                       solver='lbfgs'),
                       n_estimators=20)
    model.fit(Xs, ys_classif, Xt=Xt[:10], yt=yt_classif[:10])
    copy_model = copy.deepcopy(model)
    assert np.all(model.predict(Xt) == copy_model.predict(Xt))
    assert hex(id(model)) != hex(id(copy_model))
    
    
def test_tradaboost_multiclass():
    np.random.seed(0)
    X = np.random.randn(10, 3)
    y = np.random.choice(3, 10)
    model = TrAdaBoost(LogisticRegression(penalty=None,
                       solver='lbfgs'), Xt=X, yt=y,
                       n_estimators=20)
    model.fit(X, y)
    yp = model.predict(X)
    score = model.score(X, y)
    assert set(np.unique(yp)) == set([0,1,2])
    assert score == accuracy_score(y, yp)
    
    
def test_tradaboost_multireg():
    np.random.seed(0)
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 5)
    model = TrAdaBoostR2(LinearRegression(),
                         Xt=X, yt=y, 
                         n_estimators=20)
    model.fit(X, y)
    yp = model.predict(X)
    score = model.score(X, y)
    assert np.all(yp.shape == (10, 5))
    assert score == r2_score(y, yp)
    
    model = TwoStageTrAdaBoostR2(LinearRegression(),
                         Xt=X, yt=y, 
                         n_estimators=3,
                         n_estimators_fs=3)
    model.fit(X, y)
    yp = model.predict(X)
    score = model.score(X, y)
    assert np.all(yp.shape == (10, 5))
    assert score == r2_score(y, yp)
    
    
def test_tradaboost_above_05():
    np.random.seed(0)
    X = np.random.randn(10, 3)
    y = np.random.randn(10, 5)
    model = TrAdaBoostR2(LinearRegression(),
                         Xt=Xt[:10], yt=yt_reg[:10], 
                         n_estimators=20)
    model.fit(Xs, ys_reg)
    assert np.any(np.array(model.estimator_errors_)>0.5)
    
    model = TrAdaBoostR2(Ridge(1.),
                         Xt=Xt[:20], yt=yt_reg[:20], 
                         n_estimators=20)
    model.fit(Xs, ys_reg)
    assert np.all(np.array(model.estimator_errors_)<0.5)
    
    
def test_tradaboost_lr():
    np.random.seed(0)
    model = TrAdaBoost(LogisticRegression(penalty=None),
                         Xt=Xt[:10], yt=yt_classif[:10], 
                         n_estimators=20, lr=.1)
    model.fit(Xs, ys_classif)
    err1 = model.estimator_errors_
    
    model = TrAdaBoost(LogisticRegression(penalty=None),
                         Xt=Xt[:10], yt=yt_classif[:10], 
                         n_estimators=20, lr=2.)
    model.fit(Xs, ys_classif)
    err2 = model.estimator_errors_
    
    assert np.sum(err1) > 5 * np.sum(err2)
    
    
def test_tradaboost_sparse_matrix():
    X = scipy.sparse.csr_matrix(np.eye(200))
    y = np.random.randn(100)
    yc = np.random.choice(["e", "p"], 100)
    Xt = X[:100]
    Xs = X[100:]
    
    model = TrAdaBoost(RidgeClassifier(), Xt=Xt[:10], yt=yc[:10])
    model.fit(Xs, yc)
    model.score(Xt, yc)
    model.predict(Xs)
    
    model = TrAdaBoostR2(Ridge(), Xt=Xt[:10], yt=y[:10])
    model.fit(Xs, y)
    model.score(Xt, y)
    model.predict(Xs)
    
    model = TwoStageTrAdaBoostR2(Ridge(), Xt=Xt[:10], yt=y[:10], n_estimators=3)
    model.fit(Xs, y)
    model.score(Xt, y)
    model.predict(Xs)