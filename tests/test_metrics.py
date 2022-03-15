"""
Test base
"""

import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from adapt.base import BaseAdaptDeep, BaseAdaptEstimator
from adapt.metrics import *
from adapt.instance_based import KMM
from adapt.feature_based import CORAL

Xs = np.random.randn(100, 2)
Xt = np.random.randn(100, 2)+1.
ys = np.random.randn(100)

base_est = BaseAdaptEstimator(Xt=Xt)
base_deep = BaseAdaptDeep(Xt=Xt)

base_est.fit(Xs, ys)
base_deep.fit(Xs, ys)


def test_all_metrics():
    cov_distance(Xs, Xt)
    frechet_distance(Xs, Xt)
    linear_discrepancy(Xs, Xt)
    normalized_linear_discrepancy(Xs, Xt)
    normalized_frechet_distance(Xs, Xt)
    j_score(Xs, Xt)
    domain_classifier(Xs, Xt)
    domain_classifier(Xs, Xt, LogisticRegression())
    reverse_validation(base_est, Xs, ys, Xt)
    reverse_validation(base_deep, Xs, ys, Xt)
    
    
def test_adapt_scorer():
    if os.name != 'nt':
        scorer = make_uda_scorer(j_score, Xs, Xt)
        adapt_model = KMM(LinearRegression(), Xt=Xt, kernel="rbf", gamma=0.)
        gs = GridSearchCV(adapt_model, {"gamma": [1000, 1e-5]},
                          scoring=scorer, return_train_score=True,
                          cv=3, verbose=0, refit=False)
        gs.fit(Xs, ys)
        assert gs.cv_results_['mean_train_score'].argmax() == 0

        scorer = make_uda_scorer(cov_distance, Xs, Xt)
        adapt_model = CORAL(LinearRegression(), Xt=Xt, lambda_=1.)
        gs = GridSearchCV(adapt_model, {"lambda_": [1e-5, 10000.]},
                          scoring=scorer, return_train_score=True,
                          cv=3, verbose=0, refit=False)
        gs.fit(Xs, ys)
        assert gs.cv_results_['mean_train_score'].argmax() == 0
        assert gs.cv_results_['mean_test_score'].argmax() == 0