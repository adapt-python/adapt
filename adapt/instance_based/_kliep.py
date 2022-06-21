"""
Kullback-Leibler Importance Estimation Procedure
"""
import itertools
import warnings

import numpy as np
from sklearn.metrics import pairwise
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.metrics.pairwise import KERNEL_PARAMS

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed

EPS = np.finfo(float).eps


@make_insert_doc()
class KLIEP(BaseAdaptEstimator):
    """
    KLIEP: Kullback–Leibler Importance Estimation Procedure
    
    KLIEP is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a source instances **reweighting** which minimizes the 
    **Kullback-Leibler divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\alpha_i K(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data.
    - :math:`\\alpha_i` are the basis functions coefficients.
    - :math:`K(x, x_i) = \\text{exp}(-\\gamma ||x - x_i||^2)`
      for instance if ``kernel="rbf"``.
      
    KLIEP algorithm consists in finding the optimal :math:`\\alpha_i` according to
    the following optimization problem:
    
    .. math::
    
        \max_{\\alpha_i } \sum_{x_j \in X_T} \log(
        \sum_{x_i \in X_T} \\alpha_i K(x_j, x_i))
        
    Subject to:
    
    .. math::
    
        \sum_{x_j \in X_S} \sum_{x_i \in X_T} \\alpha_i K(x_j, x_i)) = n_S
        
    Where:
    
    - :math:`X_T` is the source input data of size :math:`n_S`.
    
    The above OP is solved through gradient ascent algorithm.
    
    Furthemore a LCV procedure can be added to select the appropriate
    parameters of the kernel function :math:`K` (typically, the paramter
    :math:`\\gamma` of the Gaussian kernel). The parameter is then selected using
    cross-validation on the :math:`J` score defined as follows:
    :math:`J = \\frac{1}{|\\mathcal{X}|} \\sum_{x \\in \\mathcal{X}} \\text{log}(w(x))`
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    KLIEP method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
    Parameters
    ----------
    kernel : str (default="rbf")
        Kernel metric.
        Possible values: [‘additive_chi2’, ‘chi2’,
        ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
        ‘laplacian’, ‘sigmoid’, ‘cosine’]
    
    sigmas : float or list of float (default=None)
        Deprecated, please use the ``gamma`` parameter
        instead. (See below).
        
    cv : int (default=5)
        Cross-validation split parameter.
        Used only if sigmas has more than one value.
        
    max_centers : int (default=100)
        Maximal number of target instances use to
        compute kernels.
        
    lr : float (default=1e-4)
        Learning rate of the gradient ascent.
        
    tol : float (default=1e-6)
        Optimization threshold.
        
    max_iter : int (default=5000)
        Maximal iteration of the gradient ascent
        optimization.
        
    Yields
    ------
    gamma : float or list of float
        Kernel parameter ``gamma``.
        
        - For kernel = chi2::
        
            k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

        - For kernel = poly or polynomial::
        
            K(X, Y) = (gamma <X, Y> + coef0)^degree
            
        - For kernel = rbf::
        
            K(x, y) = exp(-gamma ||x-y||^2)
        
        - For kernel = laplacian::
        
            K(x, y) = exp(-gamma ||x-y||_1)
        
        - For kernel = sigmoid::
        
            K(X, Y) = tanh(gamma <X, Y> + coef0)
            
        If a list is given, the LCV process is performed to
        select the best parameter ``gamma``.
        
    coef0 : floaf or list of float
        Kernel parameter ``coef0``.
        Used for ploynomial and sigmoid kernels.
        See ``gamma`` parameter above for the 
        kernel formulas.
        If a list is given, the LCV process is performed to
        select the best parameter ``coef0``.
        
    degree : int or list of int
        Degree parameter for the polynomial
        kernel. (see formula in the ``gamma``
        parameter description).
        If a list is given, the LCV process is performed to
        select the best parameter ``degree``.

    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
        
    best_params_ : float
        Best kernel params combination
        deduced from the LCV procedure.
        
    alphas_ : numpy array
        Basis functions coefficients.
        
    centers_ : numpy array
        Center points for kernels.
        
    j_scores_ : dict
        dict of J scores with the
        kernel params combination as
        keys and the J scores as values.
        
    estimator_ : object
        Fitted estimator.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import KLIEP
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = KLIEP(RidgeClassifier(), Xt=Xt, kernel="rbf", gamma=[0.1, 1.], random_state=0)
    >>> model.fit(Xs, ys)
    Fit weights...
    Cross Validation process...
    Parameter {'gamma': 0.1} -- J-score = 0.013 (0.003)
    Parameter {'gamma': 1.0} -- J-score = 0.120 (0.026)
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.85

    See also
    --------
    KMM

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3248-direct-importance-estimation\
-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf>`_ \
M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe. \
"Direct importance estimation with model selection and its application \
to covariateshift adaptation". In NIPS 2007
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 kernel="rbf",
                 sigmas=None,
                 max_centers=100,
                 cv=5,
                 lr=1e-4,
                 tol=1e-6,
                 max_iter=5000,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        if sigmas is not None:
            warnings.warn("The `sigmas` argument is deprecated, "
              "please use the `gamma` argument instead.",
              DeprecationWarning)
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_weights(self, Xs, Xt, **kwargs):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        kwargs : key, value argument
            Not used, present here for adapt consistency.
            
        Returns
        -------
        weights_ : sample weights
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        self.j_scores_ = {}
        
        # LCV GridSearch
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        
        # Handle deprecated sigmas (will be removed)
        if (self.sigmas is not None) and (not "gamma" in kernel_params):
            kernel_params["gamma"] = self.sigmas
        
        params_dict = {k: (v if hasattr(v, "__iter__") else [v]) for k, v in kernel_params.items()}
        options = params_dict
        keys = options.keys()
        values = (options[key] for key in keys)
        params_comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

        if len(params_comb) > 1:
            # Cross-validation process   
            if len(Xt) < self.cv:
                raise ValueError("Length of Xt is smaller than cv value")

            if self.verbose:
                print("Cross Validation process...")

            shuffled_index = np.arange(len(Xt))
            np.random.shuffle(shuffled_index)
            
            max_ = -np.inf
            for params in params_comb:
                cv_scores = self._cross_val_jscore(Xs, Xt[shuffled_index], params, self.cv)
                self.j_scores_[str(params)] = np.mean(cv_scores)

                if self.verbose:
                    print("Parameters %s -- J-score = %.3f (%.3f)"%
                          (str(params), np.mean(cv_scores), np.std(cv_scores)))

                if self.j_scores_[str(params)] > max_:
                    self.best_params_ = params
                    max_ = self.j_scores_[str(params)]
        else:
            self.best_params_ = params_comb[0]

        self.alphas_, self.centers_ = self._fit(Xs, Xt, self.best_params_)

        self.weights_ = np.dot(
            pairwise.pairwise_kernels(Xs, self.centers_,
                                     metric=self.kernel,
                                     **self.best_params_),
            self.alphas_
            ).ravel()
        return self.weights_
    
    
    def predict_weights(self, X=None):
        """
        Return fitted source weights
        
        If ``None``, the fitted source weights are returned.
        Else, sample weights are computing using the fitted
        ``alphas_`` and the chosen ``centers_``.
        
        Parameters
        ----------
        X : array (default=None)
            Input data.
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            if X is None or not hasattr(self, "alphas_"):
                return self.weights_
            else:
                X = check_array(X)
                weights = np.dot(
                pairwise.pairwise_kernels(X, self.centers_,
                                         metric=self.kernel,
                                         **self.best_params_),
                self.alphas_
                ).ravel()
                return weights
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")


    def _fit(self, Xs, Xt, kernel_params):
        index_centers = np.random.choice(
                        len(Xt),
                        min(len(Xt), self.max_centers),
                        replace=False)
        centers = Xt[index_centers]
        
        A = pairwise.pairwise_kernels(Xt, centers, metric=self.kernel,
                                      **kernel_params)
        B = pairwise.pairwise_kernels(centers, Xs, metric=self.kernel,
                                      **kernel_params)
        b = np.mean(B, axis=1)
        b = b.reshape(-1, 1)

        alpha = np.ones((len(centers), 1)) / len(centers)
        previous_objective = -np.inf
        objective = np.mean(np.log(np.dot(A, alpha) + EPS))
        if self.verbose > 1:
                print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
        k = 0
        while k < self.max_iter and objective-previous_objective > self.tol:
            previous_objective = objective
            alpha_p = np.copy(alpha)
            alpha += self.lr * np.dot(
                np.transpose(A), 1./(np.dot(A, alpha) + EPS)
            )
            alpha += b * ((((1-np.dot(np.transpose(b), alpha)) /
                            (np.dot(np.transpose(b), b) + EPS))))
            alpha = np.maximum(0, alpha)
            alpha /= (np.dot(np.transpose(b), alpha) + EPS)
            objective = np.mean(np.log(np.dot(A, alpha) + EPS))
            k += 1
            
            if self.verbose > 1:
                if k%100 == 0:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))

        return alpha, centers


    def _cross_val_jscore(self, Xs, Xt, kernel_params, cv):        
        split = int(len(Xt) / cv)
        cv_scores = []
        for i in range(cv):
            test_index = np.arange(i * split, (i + 1) * split)
            train_index = np.array(
                list(set(np.arange(len(Xt))) - set(test_index))
            )

            alphas, centers = self._fit(Xs,
                                        Xt[train_index],
                                        kernel_params)

            j_score = np.mean(np.log(
                np.dot(
                    pairwise.pairwise_kernels(Xt[test_index],
                                             centers,
                                             metric=self.kernel,
                                             **kernel_params),
                    alphas
                ) + EPS
            ))
            cv_scores.append(j_score)
        return cv_scores
