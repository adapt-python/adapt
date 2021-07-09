"""
Kullback-Leibler Importance Estimation Procedure
"""

import inspect
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise
from sklearn.exceptions import NotFittedError
import tensorflow as tf

from adapt.utils import check_arrays, check_one_array, check_estimator

EPS = np.finfo(float).eps

class KLIEP:
    """
    KLIEP: Kullback–Leibler Importance Estimation Procedure
    
    KLIEP is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a source instances **reweighting** which minimizes the 
    **Kullback-Leibler divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\alpha_i K_{\sigma}(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data.
    - :math:`\\alpha_i` are the basis functions coefficients.
    - :math:`K_{\sigma}(x, x_i) = \\text{exp}(-\\frac{||x - x_i||^2}{2\sigma^2})`
      are kernel functions of bandwidth :math:`\sigma`.
      
    KLIEP algorithm consists in finding the optimal :math:`\\alpha_i` according to
    the following optimization problem:
    
    .. math::
    
        \max_{\\alpha_i } \sum_{x_i \in X_T} \log(
        \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_i))
        
    Subject to:
    
    .. math::
    
        \sum_{x_k \in X_S} \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_k)) = n_S
        
    Where:
    
    - :math:`X_T` is the source input data of size :math:`n_S`.
    
    The above OP is solved through gradient ascent algorithm.
    
    Furthemore a LCV procedure can be added to select the appropriate
    bandwidth :math:`\sigma`. The parameter is then selected using
    cross-validation on the :math:`J` score defined as follow:
    :math:`J = \\frac{1}{|\\mathcal{X}|} \\sum_{x \\in \\mathcal{X}} \\text{log}(w(x))`
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    KLIEP method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task. 
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.
        
    sigmas : float or list of float (default=1/nb_features)
        Kernel bandwidths.
        If ``sigmas`` is a list of multiple values, the
        kernel bandwidth is selected with the LCV procedure.
        
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
        
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
        
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.

    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
        
    sigma_ : float
        Sigma selected for the kernel
        
    alphas_ : numpy array
        Basis functions coefficients.
        
    centers_ : numpy array
        Center points for kernels.
        
    self.j_scores_ : list of float
        List of J scores.
        
    estimator_ : object
        Fitted estimator.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.instance_based import KLIEP
    >>> np.random.seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = np.array([-0.2 * x if x<0.5 else 1. for x in Xs])
    >>> yt = -0.2 * Xt
    >>> kliep = KLIEP(sigmas=[0.1, 1, 10], random_state=0)
    >>> kliep.fit_estimator(Xs, ys)
    >>> np.abs(kliep.predict(Xt).ravel() - yt).mean()
    0.09388...
    >>> kliep.fit(Xs, ys, Xt)
    Fitting weights...
    Cross Validation process...
    Parameter sigma = 0.1000 -- J-score = 0.059 (0.001)
    Parameter sigma = 1.0000 -- J-score = 0.427 (0.003)
    Parameter sigma = 10.0000 -- J-score = 0.704 (0.017)
    Fitting estimator...
    >>> np.abs(kliep.predict(Xt).ravel() - yt).mean()
    0.00302...

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
                 sigmas=None,
                 max_centers=100,
                 cv=5,
                 lr=1e-4,
                 tol=1e-6,
                 max_iter=5000,
                 copy=True,
                 verbose=1,
                 random_state=None):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.estimator_ = check_estimator(estimator, copy=copy)
        self.sigmas = sigmas
        self.cv = cv
        self.max_centers = max_centers
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state


    def fit_weights(self, Xs, Xt):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        Returns
        -------
        weights_ : sample weights
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        Xs = check_one_array(Xs)
        Xt = check_one_array(Xt)
        
        self.j_scores_ = []

        if hasattr(self.sigmas, "__iter__"):
            # Cross-validation process   
            if len(Xt) < self.cv:
                raise ValueError("Length of Xt is smaller than cv value")

            if self.verbose:
                print("Cross Validation process...")

            shuffled_index = np.arange(len(Xt))
            np.random.shuffle(shuffled_index)

            for sigma in self.sigmas:
                cv_scores = self._cross_val_jscore(Xs, Xt[shuffled_index], sigma, self.cv)
                self.j_scores_.append(np.mean(cv_scores))

                if self.verbose:
                    print("Parameter sigma = %.4f -- J-score = %.3f (%.3f)"%
                          (sigma, np.mean(cv_scores), np.std(cv_scores)))

            self.sigma_ = self.sigmas[np.argmax(self.j_scores_)]                
        else:
            self.sigma_ = self.sigmas

        self.alphas_, self.centers_ = self._fit(Xs, Xt, self.sigma_)

        self.weights_ = np.dot(
            pairwise.rbf_kernel(Xs, self.centers_, self.sigma_),
            self.alphas_
            ).ravel()
        return self.weights_
    
    
    def fit_estimator(self, X, y,
                      sample_weight=None,
                      **fit_params):
        """
        Fit estimator.
        
        Parameters
        ----------
        X : array
            Input data.
            
        y : array
            Output data.
            
        sample_weight : array
            Importance weighting.
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.
            
        Returns
        -------
        estimator_ : fitted estimator
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        X = check_one_array(X)
        y = check_one_array(y)
             
        if "sample_weight" in inspect.signature(self.estimator_.fit).parameters:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator_.fit(X, y, 
                                    sample_weight=sample_weight,
                                    **fit_params)
        else:
            if sample_weight is not None:
                sample_weight /= (sample_weight.sum() + EPS)
            bootstrap_index = np.random.choice(
            len(X), size=len(X), replace=True,
            p=sample_weight)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator_.fit(X[bootstrap_index], y[bootstrap_index],
                                    **fit_params)
        return self.estimator_
    

    def fit(self, Xs, ys, Xt, **fit_params):
        """
        Fit KLIEP.

        Parameters
        ----------
        Xs : numpy array
            Source input data.

        ys : numpy array
            Source output data.

        Xt : numpy array
            Target input data.

        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.

        Returns
        -------
        self : returns an instance of self
        """
        Xs, ys, Xt, _ = check_arrays(Xs, ys, Xt, None)        
        if self.verbose:
            print("Fitting weights...")
        self.fit_weights(Xs, Xt)
        if self.verbose:
            print("Fitting estimator...")
        self.fit_estimator(Xs, ys,
                           sample_weight=self.weights_,
                           **fit_params)
        return self


    def _fit(self, Xs, Xt, sigma):
        index_centers = np.random.choice(
                        len(Xt),
                        min(len(Xt), self.max_centers),
                        replace=False)
        centers = Xt[index_centers]

        A = pairwise.rbf_kernel(Xt, centers, sigma)
        b = np.mean(pairwise.rbf_kernel(centers, Xs, sigma), axis=1)
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
            alpha /= np.dot(np.transpose(b), alpha)
            objective = np.mean(np.log(np.dot(A, alpha) + EPS))
            k += 1
            
            if self.verbose > 1:
                if k%100 == 0:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))

        return alpha, centers


    def _cross_val_jscore(self, Xs, Xt, sigma, cv):
        split = int(len(Xt) / cv)
        cv_scores = []
        for i in range(cv):
            test_index = np.arange(i * split, (i + 1) * split)
            train_index = np.array(
                list(set(np.arange(len(Xt))) - set(test_index))
            )

            alphas, centers = self._fit(Xs,
                                        Xt[train_index],
                                        sigma)

            j_score = np.mean(np.log(
                np.dot(
                    pairwise.rbf_kernel(Xt[test_index],
                                        centers,
                                        sigma),
                    alphas
                )
            ))
            cv_scores.append(j_score)
        return cv_scores


    def predict(self, X):
        """
        Return estimator predictions.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of estimator.
        """        
        X = check_one_array(X)
        return self.estimator_.predict(X)


    def predict_weights(self, X=None):
        """
        Return fitted source weights
        
        If ``None``, the fitted source weights are returned.
        Else, sample weights are computing using the fitted
        ``alphas_`` and the chosen ``centers_`.
        
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
                X = check_one_array(X)
                weights = np.dot(
                pairwise.rbf_kernel(X, self.centers_, self.sigma_),
                self.alphas_
                ).ravel()
                return weights
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")
