"""
Test functions for wann module.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from adapt.instance_based import WANN

np.random.seed(0)
Xs = np.concatenate((
    np.random.randn(50)*0.1,
    np.random.randn(50)*0.1 + 1.,
)).reshape(-1, 1)
Xt = (np.random.randn(100) * 0.1).reshape(-1, 1)
ys = np.array([0.2 * x if x<0.5
               else 10 for x in Xs.ravel()]).reshape(-1, 1)
yt = np.array([0.2 * x if x<0.5
               else 10 for x in Xt.ravel()]).reshape(-1, 1)

def test_fit():
    np.random.seed(0)
    tf.random.set_seed(0)
    model = WANN(random_state=0, optimizer=Adam(0.01))
    model.fit(Xs, ys, Xt, yt, epochs=200, verbose=0)
    assert np.abs(model.predict(Xt) - yt).sum() < 10
