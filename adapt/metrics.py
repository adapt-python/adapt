import inspect
import copy

import numpy as np
import tensorflow as tf
from scipy import linalg
from sklearn.metrics import pairwise
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from adapt.utils import get_default_discriminator
from tensorflow.keras.optimizers import Adam

EPS = np.finfo(float).eps


def _fit_alpha(Xs, Xt, centers, sigma):
    """
    Fit alpha coeficients to compute J-score
    """
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
    """
    Compute the mean absolute difference
    between the covariance matrixes of Xs and Xt
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    Returns
    -------
    score : float
    
    See also
    --------
    frechet_distance
    CORAL

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1511.05547.pdf>`_ Sun B., Feng J., Saenko K. \
"Return of frustratingly easy domain adaptation". In AAAI, 2016.
    """
    cov_Xs = np.cov(Xs, rowvar=False)
    cov_Xt = np.cov(Xt, rowvar=False)
    return np.mean(np.abs(cov_Xs-cov_Xt))


def frechet_distance(Xs, Xt):
    """
    Compute the frechet distance
    between Xs and Xt.
    
    .. math::
        
        \\Delta = ||\\mu_S - \\mu_T||_2^2 + Tr\\left(\\Sigma_S + \\Sigma_T
        - 2 (\\Sigma_S \\cdot \\Sigma_T)^{\\frac{1}{2}} \\right)
        
    Where:
    
    - :math:`\\mu_S, \\mu_T` are the mean of Xs, Xt along first axis.
    - :math:`\\Sigma_S, \\Sigma_T` are the covariance matrix of Xs, Xt.
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    Returns
    -------
    score : float
    
    See also
    --------
    normalized_frechet_distance
    linear_discrepancy
    normalized_linear_discrepancy
    
    References
    ----------
    .. [1] `[1] <https://www.sciencedirect.com/science/article/pii/00\
47259X8290077X?via%3Dihub>`_ Dowson, D. C; Landau, B. V. "The Fréchet \
distance between multivariate normal distributions". JMVA. 1982
    """
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
    """
    Compute the linear discrepancy
    between Xs and Xt.
    
    .. math::
        
        \\Delta = \\max_{u \\in \\mathbb{R}^p} u^T (X_S^T X_S - X_T^T X_T) u
        
    Where:
    
    - :math:`p` is the number of features of Xs and Xt.
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    power_method : bool (default=False)
        Weither to use the power method
        approximation or not.
        
    n_iter : int (default=20)
        Number of iteration for power method
        
    Returns
    -------
    score : float
    
    See also
    --------
    normalized_linear_discrepancy
    frechet_distance
    normalized_frechet_distance

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0902.3430.pdf>`_ \
Y. Mansour, M. Mohri, and A. Rostamizadeh. "Domain \
adaptation: Learning bounds and algorithms". In COLT, 2009.
    """
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
    """
    Compute the normalized linear discrepancy
    between Xs and Xt.
    
    Xs and Xt are first scaled by a factor
    ``(std(Xs) + std(Xt)) / 2``
    and centered around ``(mean(Xs) + mean(Xt)) / 2``
    
    Then, the linear discrepancy is computed and divided by the number
    of features.
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    Returns
    -------
    score : float
    
    See also
    --------
    linear_discrepancy
    frechet_distance
    normalized_frechet_distance

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0902.3430.pdf>`_ \
Y. Mansour, M. Mohri, and A. Rostamizadeh. "Domain \
adaptation: Learning bounds and algorithms". In COLT, 2009.
    """
    std = (np.std(Xs) + np.std(Xt) + EPS)/2
    mu = (np.mean(Xs) + np.mean(Xt))/2
    x_max = linear_discrepancy((Xs-mu)/std, (Xt-mu)/std, power_method, n_iter)
    return x_max / Xs.shape[1]


def normalized_frechet_distance(Xs, Xt):
    """
    Compute the normalized frechet distance
    between Xs and Xt.
    
    Xs and Xt are first scaled by a factor
    ``(std(Xs) + std(Xt)) / 2``
    and centered around ``(mean(Xs) + mean(Xt)) / 2``
    
    Then, the frechet distance is computed and divided by the number
    of features.
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    Returns
    -------
    score : float
    
    See also
    --------
    frechet_distance
    linear_discrepancy
    normalized_linear_discrepancy
    
    References
    ----------
    .. [1] `[1] <https://www.sciencedirect.com/science/article/pii/00\
47259X8290077X?via%3Dihub>`_ Dowson, D. C; Landau, B. V. "The Fréchet \
distance between multivariate normal distributions". JMVA. 1982
    """
    std = (np.std(Xs) + np.std(Xt) + EPS)/2
    mu = (np.mean(Xs) + np.mean(Xt))/2
    x_max = frechet_distance((Xs-mu)/std, (Xt-mu)/std)
    return x_max / Xs.shape[1]


def j_score(Xs, Xt, max_centers=100, sigma=None):
    """
    Compute the negative J-score between Xs and Xt.
    
    .. math::
        
        \\Delta = -\\int_{\\mathcal{X}} P(X_T) \\log(P(X_T) / P(X_S))
        
    Where:
    
    - :math:`P(X_S), P(X_T)` are the probability density
      functions of Xs and Xt.
    
    The source and target probability density functions
    are approximated with a mixture of gaussian kernels
    of bandwith ``sigma`` and centered in ``max_centers``
    random points of Xt. The coefficient of the mixture
    are determined by solving a convex optimization (see [1])
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    max_centers : int (default=100)
        Maximum number of centers from Xt
        
    sigma : float (default=None)
        Kernel bandwidth. If ``None``, the mean
        of pairwise distances between data from
        Xt is used.
        
    Returns
    -------
    score : float
    
    See also
    --------
    KLIEP

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3248-direct-importance-estimation\
-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf>`_ \
M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe. \
"Direct importance estimation with model selection and its application \
to covariateshift adaptation". In NIPS 2007
    """
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
    """
    Return 1 minus the mean square error of a classifer
    disciminating between Xs and Xt.
    
    .. math::
        
        \\Delta = 1 - \\min_{h \\in H} || h(X_S) - 1 ||^2 +
        || h(X_T) ||^2
        
    Where:
    
    - :math:`H` is a class of classifier.
    
    Parameters
    ----------
    Xs : array
        Source array
        
    Xt : array
        Target array
        
    classifier : sklearn estimator or tensorflow Model instance
        Classifier
        
    fit_params : key, value arguments
        Parameters for the fit method of the classifier.
        
    Returns
    -------
    score : float
    
    See also
    --------
    reverse_validation
    DANN
        
    References
    ----------
    .. [1] `[1] <http://jmlr.org/papers/volume17/15-239/15-239.pdf>`_ Y. Ganin, \
E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, \
and V. Lempitsky. "Domain-adversarial training of neural networks". In JMLR, 2016.
    """
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
    
    y_pred = classifier.predict(X_test)
    return 1-np.mean(np.square(y_pred-y_test.reshape(y_pred.shape)))


def reverse_validation(model, Xs, ys, Xt, **fit_params):
    """
    Reverse validation.
    
    The reverse validation score is computed as a source error
    by inversing the role of the source and the target domains.
    A clone of the model is trained to adapt from the target to
    the source using the model target predictions as
    pseudo target labels. Then the final score is computed between
    the source prediction of the cloned model and the groundtruth.
    
    Parameters
    ----------
    model : BaseAdapt instance
        Adaptation model
    
    Xs : array
        Source input array
        
    ys : array
        Source output array
        
    Xt : array
        Target input array
        
    fit_params : key, value arguments
        Parameters for the fit method of the cloned model.
        
    Returns
    -------
    score : float
    
    See also
    --------
    domain_classifier
    DANN
        
    References
    ----------
    .. [1] `[1] <http://jmlr.org/papers/volume17/15-239/15-239.pdf>`_ Y. Ganin, \
E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, \
and V. Lempitsky. "Domain-adversarial training of neural networks". In JMLR, 2016.
    """
    yt = model.predict(Xt)
    
    if yt.ndim == 1 and ys.ndim > 1:
        yt = yt.reshape(-1, 1)
        
    if ys.ndim == 1 and yt.ndim > 1:
        yt = yt.ravel()
    
    clone_model = clone(model)
    clone_model.fit(Xt, yt, Xs, **fit_params)
    
    return clone_model.score_estimator(Xs, ys)