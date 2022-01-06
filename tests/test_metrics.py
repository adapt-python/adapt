"""
Test base
"""

from sklearn.linear_model import LogisticRegression
from adapt.base import BaseAdaptDeep, BaseAdaptEstimator
from adapt.metrics import *

Xs = np.random.randn(100, 2)
Xt = np.random.randn(100, 2)
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
    