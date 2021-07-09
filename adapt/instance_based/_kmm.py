"""
Kernel Mean Matching
"""

import inspect
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise
from cvxopt import matrix, solvers
import tensorflow as tf

from adapt.utils import check_arrays, check_one_array, check_estimator

EPS = np.finfo(float).eps

class KMM:
    """
    KMM: Kernel Mean Matching
    
    KMM is a sample bias correction method for domain adaptation based on the
    minimization of the **Maximum Mean Discreapancy** (MMD) between source
    and target domains.
    
    The algorithm corrects input source and taregt distributions differences by
    **reweighting** the source instances such that the means of the source and target
    instances in a **reproducing kernel Hilbert space** (RKHS) are "close".
    
    This leads to solve the following **quadratic optimization problem**:
    
    .. math::
    
        \min_{w} \\frac{1}{2} w^T K w - \kappa^T w
        
    Subject to:
    
    .. math::
    
        w_i \in [0, B] \\text{ and } |\sum_{i=1}^{n_S} w_i - n_S| \leq m \epsilon
        
    Where:
    
    - :math:`K_{ij} = k(x_i, x_j)` with :math:`x_i, x_j \in X_S`
      and :math:`k` a kernel.
    - :math:`\\kappa_{i} = \\frac{n_S}{n_T} \sum_{x_j \in X_T} k(x_i, x_j)` 
      with :math:`x_i \\in X_S`.
    - :math:`w_i` are the source instance weights.
    - :math:`X_S, X_T` are respectively the input source and target dataset.
    - :math:`B, \epsilon` are two KMM hyperparameters.
    
    After solving the above OP, an estimator is fitted using the reweighted
    labeled source instances.
    
    KMM method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.

    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task. 
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.
        
    B: float, optional (default=1000)
        Bounding weights parameter.
        
    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(len(Xs)) - 1)/np.sqrt(len(Xs))``
        with ``Xs`` the source input dataset.
        
    kernel : str, optional (default="rbf")
        Kernel metric.
        Possible values: [‘additive_chi2’, ‘chi2’,
        ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
        ‘laplacian’, ‘sigmoid’, ‘cosine’]
        
    kernel_params : dict, optional (default=None)
        Kernel additional parameters
        
    tol: float, optional (default=None)
        Optimization threshold. If ``None``
        default parameters from cvxopt are used.
        
    max_iter: int, optional (default=100)
        Maximal iteration of the optimization.
        
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
    
    estimator_ : object
        Estimator.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.instance_based import KMM
    >>> np.random.seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = np.array([-0.2 * x if x<0.5 else 1. for x in Xs])
    >>> yt = -0.2 * Xt
    >>> kmm = KMM(random_state=0)
    >>> kmm.fit_estimator(Xs, ys)
    >>> np.abs(kmm.predict(Xt).ravel() - yt).mean()
    0.09388...
    >>> kmm.fit(Xs, ys, Xt)
    Fitting weights...
     pcost       dcost       gap    pres   dres
     0:  3.7931e+04 -1.2029e+06  3e+07  4e-01  2e-15
    ...
    13: -4.9095e+03 -4.9095e+03  8e-04  2e-16  1e-16
    Optimal solution found.
    Fitting estimator...
    >>> np.abs(kmm.predict(Xt).ravel() - yt).mean()
    0.00588...

    See also
    --------
    KLIEP

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3075-correcting-sample-selection\
-bias-by-unlabeled-data.pdf>`_ J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf, \
and A. J. Smola. "Correcting sample selection bias by unlabeled data." In NIPS, 2007.
    """

    def __init__(self,
                 estimator=None,
                 B=1000,
                 epsilon=None,
                 kernel="rbf",
                 kernel_params=None,
                 tol=None,
                 max_iter=100,
                 copy=True,
                 verbose=1,
                 random_state=None):

        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.estimator_ = check_estimator(estimator, copy=copy)
        self.B = B
        self.epsilon = epsilon
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.tol = tol
        self.max_iter = max_iter
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        
        if self.kernel_params is None:
            self.kernel_params = {}


    def fit(self, Xs, ys, Xt, **fit_params):
        """
        Fit KMM.

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
                
        n_s = len(Xs)
        n_t = len(Xt)
        
        # Get epsilon
        if self.epsilon is None:
            epsilon = (np.sqrt(n_s) - 1)/np.sqrt(n_s)
        else:
            epsilon = self.epsilon

        # Compute Kernel Matrix
        K = pairwise.pairwise_kernels(Xs, Xs, metric=self.kernel,
                                      **self.kernel_params)
        K = (1/2) * (K + K.transpose())

        # Compute q
        kappa = pairwise.pairwise_kernels(Xs, Xt,
                                          metric=self.kernel,
                                          **self.kernel_params)
        kappa = (n_s/n_t) * np.dot(kappa, np.ones((n_t, 1)))
        
        P = matrix(K)
        q = -matrix(kappa)
        
        # Define constraints
        G = np.ones((2*n_s+2, n_s))
        G[1] = -G[1]
        G[2:n_s+2] = np.eye(n_s)
        G[n_s+2:n_s*2+2] = -np.eye(n_s)
        h = np.ones(2*n_s+2)
        h[0] = n_s*(1+epsilon)
        h[1] = n_s*(epsilon-1)
        h[2:n_s+2] = self.B
        h[n_s+2:] = 0

        G = matrix(G)
        h = matrix(h)
        
        solvers.options["show_progress"] = bool(self.verbose)
        solvers.options["maxiters"] = self.max_iter
        if self.tol is not None:
            solvers.options['abstol'] = self.tol
            solvers.options['reltol'] = self.tol
            solvers.options['feastol'] = self.tol
        else:
            solvers.options['abstol'] = 1e-7
            solvers.options['reltol'] = 1e-6
            solvers.options['feastol'] = 1e-7
        weights = solvers.qp(P, q, G, h)['x']

        self.weights_ = np.array(weights).ravel()
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

    
    def predict_weights(self):
        """
        Return fitted source weights
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            return self.weights_
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")
