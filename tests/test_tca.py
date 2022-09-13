import numpy as np

from adapt.metrics import normalized_linear_discrepancy
from adapt.feature_based import TCA

np.random.seed(0)
n = 50
m = 50
p = 6

Xs = np.random.randn(m, p)*0.1 + np.array([0.]*(p-2) + [2., 2.])
Xt = np.random.randn(n, p)*0.1


def test_tca():
    tca = TCA(n_components=2, kernel="rbf", gamma=0.01, random_state=0)
    Xst = tca.fit_transform(Xs, Xt)
    assert np.abs(Xst - tca.transform(Xs, "src")).sum() == 0.
    assert Xst.shape[1] == 2
    assert (normalized_linear_discrepancy(Xs, Xt) >
            2 * normalized_linear_discrepancy(Xst, tca.transform(Xt)))