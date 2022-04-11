import numpy as np
import os

from adapt.instance_based import LDM

np.random.seed(0)
n = 50
m = 50
p = 6

Xs = np.concatenate((np.random.randn(int(m/2), p)*0.1,
                     2+np.random.randn(int(m/2), p)*0.1))
Xt = np.random.randn(n, p)*0.1
ys = Xs[:,0]
ys[Xs[:,0]>1] = 2.
yt = Xt[:,0]


def test_ldm():
    if os.name != 'nt':
        ldm = LDM()
        weights = ldm.fit_weights(Xs, Xt)
        ldm.fit(Xs, ys, Xt)
        yp = ldm.predict(Xt)
        assert ldm.score(Xt, yt) > 0.9
        assert weights[:25].mean() > 10 * weights[25:].mean()
    
    
def test_ldm_diff_size():
    if os.name != 'nt':
        ldm = LDM()
        weights = ldm.fit_weights(Xs, Xt[:40])
        assert weights[:25].mean() > 10 * weights[25:].mean()