import inspect
import copy

import numpy as np
import tensorflow as tf
from scipy import linalg
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from adapt.utils import get_default_discriminator
from tensorflow.keras.optimizers import Adam

EPS = np.finfo(float).eps


def make_target_scorer(score_func, Xt, yt,
                       Xs=None,
                       ys=None,
                       greater_is_better=True,
                       **fit_params):
    
    def scorer(estimator, X, y_true=None):
        
        if Xs is not None and ys is not None:
            if hasattr(estimator.model, "predict_features"):
                args = inspect.getfullargspec(estimator.model.predict_features).args
                if "domain" in args:
                    Xs_enc = estimator.model.predict_features(Xs, domain="src")
                else:
                    Xs_enc = estimator.model.predict_features(Xs)
                estimator.model.fit_estimator(Xs_enc, ys, **fit_params)
            elif hasattr(estimator.model, "predict_weights"):
                sample_weight = estimator.model.predict_weights()
                estimator.model.fit_estimator(Xs, ys, sample_weight=sample_weight, **fit_params)
            else:
                raise ValueError("Invalid estimator")
            
        y_pred = estimator.predict(Xt)
        score = score_func(yt, y_pred)
    
        if not greater_is_better:
            score *= -1
        return score
    return scorer


def make_adapt_scorer(score_func,
                      Xs=None,
                      ys=None,
                      Xt=None,
                      yt=None,
                      greater_is_better=True,
                      **kwargs):
    
    def scorer(estimator, X, y_true=None):
        
        if Xt is not None and yt is not None:
            y_pred = estimator.predict(Xt)
            score = score_func(yt, y_pred, **kwargs)        
        else:
            if y_true is not None and Xt is None:
                y_pred = estimator.predict(X)
                score = score_func(y_true, y_pred, **kwargs)

            else:
                if hasattr(estimator.model, "predict_features"):
                    args = inspect.getfullargspec(estimator.model.predict_features).args
                    if "domain" in args:
                        X_tgt = estimator.model.predict_features(X, domain="tgt")
                        X_src = estimator.model.predict_features(Xs, domain="src")
                    else:
                        X_tgt = estimator.model.predict_features(X)
                        X_src = estimator.model.predict_features(Xs)
                    X_src = check_one_array(X_src)
                    X_tgt = check_one_array(X_tgt)
                    score = score_func(X_src, X_tgt, **kwargs)

                elif hasattr(estimator.model, "predict_weights"):
                    # TODO if in TrAda or KMM Xs and Xt have same size,
                    # an error will be raised
                    sample_weight = estimator.model.predict_weights().ravel()
                    
                    if len(sample_weight) != len(Xs):
                        sample_weight = estimator.model.predict_weights(Xs).ravel()
                    
                    if sample_weight.sum() <= 0:
                        sample_weight = np.ones(len(sample_weight))
                    sample_weight /= sample_weight.sum()
                    bootstrap_index = np.random.choice(
                    len(Xs), size=len(Xs), replace=True, p=sample_weight)
                                       
                    X_src = Xs[bootstrap_index]  
                    X_src = check_one_array(X_src)
                    
                    if Xt is not None:
                        X_tgt = check_one_array(Xt)                   
                        score = score_func(X_src, X_tgt, **kwargs)
                    else:
                        X = check_one_array(X)                   
                        score = score_func(X_src, X, **kwargs)

                else:
                    raise ValueError("Invalid estimator")
        if not greater_is_better:
            score *= -1
        return score
    
    return scorer


def _fit_alpha(Xs, Xt, centers, sigma):
    A = pairwise.rbf_kernel(Xt, centers, sigma)
    b = np.mean(pairwise.rbf_kernel(centers, Xs, sigma), axis=1)
    b = b.reshape(-1, 1)

    alpha = np.ones((len(centers), 1)) / len(centers)
    previous_objective = -np.inf
    objective = np.mean(np.log(np.dot(A, alpha) + EPS))

    k = 0
    while k < 5000 and objective-previous_objective > 1e-6:
        previous_objective = objective
        alpha_p = np.copy(alpha)
        alpha += 1e-4 * np.dot(
            np.transpose(A), 1./(np.dot(A, alpha) + EPS)
        )
        alpha += b * ((((1-np.dot(np.transpose(b), alpha)) /
                        (np.dot(np.transpose(b), b) + EPS))))
        alpha = np.maximum(0, alpha)
        alpha /= (np.dot(np.transpose(b), alpha) + EPS)
        objective = np.mean(np.log(np.dot(A, alpha) + EPS))
        k += 1

    return alpha


def cov_distance(Xs, Xt):
    cov_Xs = np.cov(Xs, rowvar=False)
    cov_Xt = np.cov(Xt, rowvar=False)
    return np.mean(np.abs(cov_Xs-cov_Xt))


def frechet_distance(Xs, Xt):
    mu1 = np.mean(Xs, axis=0)    
    sigma1 = np.cov(Xs, rowvar=False)
    mu2 = np.mean(Xt, axis=0)
    sigma2 = np.cov(Xt, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    product = np.array(sigma1.dot(sigma2))
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def linear_discrepancy(Xs, Xt, power_method=False, n_iter=20):
    M = (1/len(Xs)) * np.dot(np.transpose(Xs), Xs) - (1/len(Xt)) * np.dot(np.transpose(Xt), Xt)
    if power_method:
        x = np.ones(len(M))
        for _ in range(n_iter):
            x = M.dot(x)
            x_max = np.max(np.abs(x))
            x = (1 / (x_max + EPS)) * x
    else:
        e, v = linalg.eig(M)
        x_max = np.max(np.abs(e))
    return x_max


def normalized_linear_discrepancy(Xs, Xt, power_method=False, n_iter=20):
    std = (np.std(Xs) + np.std(Xt) + EPS)/2
    mu = (np.mean(Xs) + np.mean(Xt))/2
    x_max = linear_discrepancy((Xs-mu)/std, (Xt-mu)/std, power_method, n_iter)
    return x_max / Xs.shape[1]


def normalized_frechet_distance(Xs, Xt):
    std = (np.std(Xs) + np.std(Xt) + EPS)/2
    mu = (np.mean(Xs) + np.mean(Xt))/2
    x_max = frechet_distance((Xs-mu)/std, (Xt-mu)/std)
    return x_max / Xs.shape[1]


def j_score(Xs, Xt, max_centers=100, sigma=None):    
    if len(Xt) > max_centers:
        random_index = np.random.choice(
        len(Xt), size=max_centers, replace=False)
        centers = Xt[random_index]
    else:
        centers = Xt
        
    if sigma is None:
        sigma = pairwise.euclidean_distances(Xt, Xt).mean()
    
    alphas = _fit_alpha(Xs, Xt, centers, sigma)
    
    j_score_ = np.mean(np.log(np.dot(
        pairwise.rbf_kernel(Xt,
                            centers,
                            sigma),
        alphas) + EPS))
    return -j_score_


def domain_classifier(Xs, Xt, classifier=None, **fit_params):
    Xs_train, Xs_test = train_test_split(Xs, train_size=0.8)
    Xt_train, Xt_test = train_test_split(Xt, train_size=0.8)
    
    X_train = np.concatenate((Xs_train, Xt_train))
    y_train = np.concatenate((np.zeros(len(Xs_train)),
                              np.ones(len(Xt_train))))
    X_test = np.concatenate((Xs_test, Xt_test))
    y_test = np.concatenate((np.zeros(len(Xs_test)),
                              np.ones(len(Xt_test))))
    
    if classifier is None:
        classifier = get_default_discriminator()
        classifier.compile(optimizer=Adam(0.001), loss="bce")
        if fit_params == {}:
            fit_params = dict(epochs=max(1, int(3000 * 64 / len(X_train))),
                              batch_size=64,
                              verbose=0)
    classifier.fit(X_train, y_train, **fit_params)
    
    y_pred = classifier(tf.identity(X_test))
    return 1-np.mean(np.square(y_pred-y_test.reshape(y_pred.shape)))