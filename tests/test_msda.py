"""
Test functions for fe module.
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapt.feature_based import mSDA
from adapt.utils import toy_classification


X, y, src_index, tgt_index, tgt_index_labeled = toy_classification()

def _get_encoder(input_shape):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_decoder(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    input_shape=input_shape))
    model.compile(loss="mse", optimizer="adam")
    return model


def test_setup():
    lr = LogisticRegression(penalty='none', solver='lbfgs')
    lr.fit(X[src_index], y[src_index])
    assert (lr.predict(X[tgt_index]) - y[tgt_index]).sum() > 20
    assert np.abs(lr.predict(X[src_index]) - y[src_index]).sum() < 5


def test_fit():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = mSDA(_get_encoder, _get_decoder,
                 LogisticRegression,
                 est_params=dict(penalty='none',
                                 solver='lbfgs'),
                 optimizer=Adam(0.01))
    model.fit(X, y, src_index, tgt_index,
              fit_params_ae=dict(epochs=800,
                                 batch_size=200,
                                 verbose=0)
             )
    assert isinstance(model.estimator_, LogisticRegression)
    assert np.abs(model.predict(X[tgt_index]) - y[tgt_index]).sum() < 10
    assert np.abs(model.predict(X[src_index]) - y[src_index]).sum() < 10
