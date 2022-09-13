from sklearn.linear_model import RidgeClassifier
from adapt.utils import make_classification_da
from adapt.instance_based import BalancedWeighting

Xs, ys, Xt, yt = make_classification_da()

def test_good_ratio():
    model = BalancedWeighting(RidgeClassifier(), gamma=0.5, Xt=Xt[:3], yt=yt[:3],
                              verbose=0, random_state=0)
    model.fit(Xs, ys)
    model.predict(Xt)
    assert model.score(Xt, yt) > 0.9
    
    
def test_bad_ratio():
    model = BalancedWeighting(RidgeClassifier(), gamma=0.99, Xt=Xt[:3], yt=yt[:3],
                              verbose=0, random_state=0)
    model.fit(Xs, ys)
    assert model.score(Xt, yt) < 0.7
    
    model = BalancedWeighting(RidgeClassifier(), gamma=0.01, Xt=Xt[:3], yt=yt[:3],
                              verbose=0, random_state=0)
    model.fit(Xs, ys)
    assert model.score(Xt, yt) < 0.9