from sklearn.linear_model import RidgeClassifier
from adapt.utils import make_classification_da
from adapt.instance_based import ULSIF, RULSIF

Xs, ys, Xt, yt = make_classification_da()


def test_ulsif():
    model = ULSIF(RidgeClassifier(0.), Xt=Xt[:73], kernel="rbf",
                   lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], random_state=0)
    model.fit(Xs, ys);
    model.predict(Xs)
    model.score(Xt, yt)
    
    
def test_rulsif():
    model = RULSIF(RidgeClassifier(0.), Xt=Xt, kernel="rbf", alpha=0.1,
                   lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], random_state=0)
    model.fit(Xs[:73], ys[:73]);
    model.predict(Xs)
    model.score(Xt, yt)