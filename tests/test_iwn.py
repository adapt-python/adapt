"""
Test functions for iwn module.
"""

from sklearn.linear_model import RidgeClassifier
from adapt.utils import make_classification_da
from adapt.instance_based import IWN

Xs, ys, Xt, yt = make_classification_da()

def test_iwn():
    model = IWN(RidgeClassifier(0.), Xt=Xt, sigma_init=0.1, random_state=0,
                pretrain=True, pretrain__epochs=100, pretrain__verbose=0)
    model.fit(Xs, ys, epochs=100, batch_size=256, verbose=0)
    model.score(Xt, yt)
    model.predict(Xs)
    model.predict_weights(Xs)
