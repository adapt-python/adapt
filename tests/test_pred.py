from sklearn.linear_model import RidgeClassifier
from adapt.utils import make_classification_da
from adapt.feature_based import PRED

Xs, ys, Xt, yt = make_classification_da()

def test_pred():
    model = PRED(RidgeClassifier(), pretrain=True, Xt=Xt[:3], yt=yt[:3],
                              verbose=0, random_state=0)
    model.fit(Xs, ys)
    model.predict(Xt)
    model.predict(Xt, "src")
    model.score(Xt, yt, domain="src")
    model.score(Xt, yt, domain="tgt")
    
    model = PRED(RidgeClassifier().fit(Xs, ys),
                 pretrain=False, Xt=Xt[:3], yt=yt[:3],
                 verbose=0, random_state=0)
    model.fit(Xs, ys)
    model.predict(Xt)
    model.predict(Xt, "src")
    model.score(Xt, yt, domain="src")
    model.score(Xt, yt, domain="tgt")