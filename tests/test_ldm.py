import numpy as np

from adapt.instance_based import LDM

np.random.seed(0)
n = 50
m = 50
p = 6

Xs = np.concatenate((np.random.randn(int(m/2), p)*0.1,
                     2+np.random.randn(int(m/2), p)*0.1))
Xt = np.random.randn(n, p)*0.1


def test_ldm():
    ldm = LDM()
    weights = ldm.fit_weights(Xs, Xt)
    assert weights[:25].mean() > 10 * weights[25:].mean()