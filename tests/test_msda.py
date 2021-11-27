"""
Test functions for fe module.
"""


import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapt.feature_based import mSDA
from adapt.utils import make_classification_da

Xs, ys, Xt, yt = make_classification_da()

def _get_encoder(input_shape=(2,)):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_decoder(input_shape=(1,), output_shape=(2,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    input_shape=input_shape))
    model.compile(loss="mse", optimizer="adam")
    return model


def test_setup():
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(Xs, ys)
    assert (lr.predict(Xt) - yt).sum() > 20
    assert np.abs(lr.predict(Xs) - ys).sum() < 5


def test_fit():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = mSDA(_get_encoder(), _get_decoder(),
                 estimator=LogisticRegression(penalty='none',
                                 solver='lbfgs'),
                 optimizer=Adam(0.01))
    model.fit(Xs, ys, Xt, epochs=800, batch_size=200, verbose=0)
    assert isinstance(model.estimator_, LogisticRegression)
    assert np.abs(model.predict(Xt) - yt).sum() < 10
    assert np.abs(model.predict(Xs) - ys).sum() < 10
    
    
def test_default():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = mSDA()
    model.fit(Xs, ys, Xt, epochs=1, batch_size=200, verbose=0)
    assert isinstance(model.encoder_.layers[1], Dense)
    
    
def test_error():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = mSDA()
    with pytest.raises(ValueError) as excinfo:
        model.fit(Xs, ys, yt.reshape(-1, 1),
                  epochs=1, batch_size=200, verbose=0)
    assert ("Xs and Xt should have same dim, got " 
            "(2,) and (1,)" in str(excinfo.value))
