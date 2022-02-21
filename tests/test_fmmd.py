import numpy as np

from adapt.feature_based import fMMD

np.random.seed(0)
n = 50
m = 50
p = 6

Xs = np.random.randn(m, p)*0.1 + np.array([0.]*(p-2) + [2., 2.])
Xt = np.random.randn(n, p)*0.1


def test_fmmd():
    fmmd = fMMD()
    fmmd.fit_transform(Xs, Xt);
    assert fmmd.features_scores_[-2:].sum() > 10 * fmmd.features_scores_[:-2].sum()
    assert np.all(fmmd.selected_features_ == [True]*4 + [False]*2)
    assert np.abs(fmmd.transform(Xs) - Xs[:, :4]).sum() == 0.

    fmmd.set_params(kernel="rbf")
    fmmd.fit_transform(Xs, Xt);
    assert fmmd.features_scores_[-2:].sum() > 10 * fmmd.features_scores_[:-2].sum()

    fmmd.set_params(kernel="poly", degree=2, gamma=0.1)
    fmmd.fit_transform(Xs, Xt);
    assert fmmd.features_scores_[-2:].sum() > 10 * fmmd.features_scores_[:-2].sum()