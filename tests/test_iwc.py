"""
Test functions for iwn module.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from adapt.utils import make_classification_da
from adapt.instance_based import IWC
from adapt.utils import get_default_discriminator
try:
    from tensorflow.keras.optimizers.legacy import Adam
except:
    from tensorflow.keras.optimizers import Adam

Xs, ys, Xt, yt = make_classification_da()

def test_iwn():
    model = IWC(RidgeClassifier(0.), classifier=RidgeClassifier(0.),
                Xt=Xt, random_state=0)
    model.fit(Xs, ys);
    model.predict(Xt)
    model.score(Xt, yt)
    w1 = model.predict_weights()
    w2 = model.predict_weights(Xs)
    assert np.abs(w1-w2).sum() < 10**-5
    
    
def test_default_classif():
    model = IWC(RidgeClassifier(0.), classifier=None,
                Xt=Xt, random_state=0)
    model.fit(Xs, ys);
    model.predict(Xt)
    model.score(Xt, yt)
    w1 = model.predict_weights()
    w2 = model.predict_weights(Xs)
    assert np.abs(w1-w2).sum() < 10**-5
    
    
def test_nn_classif():
    model = IWC(RidgeClassifier(0.), classifier=get_default_discriminator(),
                cl_params=dict(epochs=10, optimizer=Adam(), loss="bce", verbose=0),
                Xt=Xt, random_state=0)
    model.fit(Xs, ys);
    model.predict(Xt)
    model.score(Xt, yt)
    w1 = model.predict_weights()
    w2 = model.predict_weights(Xs)
    assert np.abs(w1-w2).sum() < 10**-5
    
