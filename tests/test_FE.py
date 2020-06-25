import numpy as np
import pytest
from adapt import FE

class DummyEstimator(object):
    
    def __init__(self):
        pass
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.zeros(len(X))


def test_fit_predict():
    model = FE(DummyEstimator)
    model.fit(np.ones((100,10)), np.ones(100),
              index=[range(80), range(80, 100)])
    y_pred = model.predict(np.ones((100,10)))
    assert np.all(y_pred == np.zeros(100))