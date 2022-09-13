from sklearn.linear_model import Ridge
from adapt.utils import make_regression_da
from adapt.parameter_based import LinInt

Xs, ys, Xt, yt = make_regression_da()

def test_linint():
    model = LinInt(Ridge(), Xt=Xt[:6], yt=yt[:6],              
                 verbose=0, random_state=0)
    model.fit(Xs, ys)
    model.fit(Xs, ys, Xt[:6], yt[:6])
    model.predict(Xt)
    model.score(Xt, yt)