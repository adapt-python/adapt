"""
Kernel Mean Matching
"""

import numpy as np
from sklearn.metrics import pairwise
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import KERNEL_PARAMS
from cvxopt import matrix, solvers

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed

EPS = np.finfo(float).eps


@make_insert_doc()
class KMM(BaseAdaptEstimator):
    """
    KMM: Kernel Mean Matching
    
    KMM is a sample bias correction method for domain adaptation based on the
    minimization of the **Maximum Mean Discrepancy** (MMD) between source
    and target domains.
    
    The algorithm corrects the difference between the input source and target distributions by
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
    kernel : str (default="rbf")
        Kernel metric.
        Possible values: [‘additive_chi2’, ‘chi2’,
        ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
        ‘laplacian’, ‘sigmoid’, ‘cosine’]
    
    B: float (default=1000)
        Bounding weights parameter.
        
    eps: float, optional (default=None)
        Constraint parameter.
        If ``None``,  ``eps`` is set to::
        
            eps = (np.sqrt(len(Xs)) - 1)/np.sqrt(len(Xs))

        with ``Xs`` the source input dataset.
        
    max_size : int (default=1000)
        Batch computation to speed up the fitting.
        If ``len(Xs) > max_size``, KMM is applied
        successively on seperated batch
        of size lower than ``max_size``.
        
    tol: float (default=None)
        Optimization threshold. If ``None``
        default parameters from cvxopt are used.
        
    max_iter: int (default=100)
        Maximal iteration of the optimization.
        
    Yields
    ------
    gamma : float
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
        
    coef0 : floaf
        Kernel parameter ``coef0``.
        Used for ploynomial and sigmoid kernels.
        See ``gamma`` parameter above for the 
        kernel formulas.
        
    degree : int
        Degree parameter for the polynomial
        kernel. (see formula in the ``gamma``
        parameter description)
        
    Attributes
    ----------  
    weights_ : numpy array
        Training instance weights.
    
    estimator_ : object
        Estimator.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import KMM
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = KMM(RidgeClassifier(), Xt=Xt, kernel="rbf", gamma=1., verbose=0, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.76

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
                 Xt=None,
                 kernel="rbf",
                 B=1000,
                 eps=None,
                 max_size=1000,
                 tol=None,
                 max_iter=100,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):

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
        
        if len(Xs) > self.max_size:
            size = len(Xs)
            power = 0
            while size > self.max_size:
                size = size / 2
                power += 1
            split = int(len(Xs) / 2**power)
            shuffled_index = np.random.choice(len(Xs), len(Xs), replace=False)
            weights = np.zeros(len(Xs))
            for i in range(2**power):
                index = shuffled_index[split*i:split*(i+1)]
                weights[index] = self._fit_weights(Xs[index], Xt)
        else:
            weights = self._fit_weights(Xs, Xt)
        
        self.weights_ = weights
        return self.weights_
            
            
    def _fit_weights(self, Xs, Xt):
        n_s = len(Xs)
        n_t = len(Xt)
        
        # Get epsilon
        if self.eps is None:
            eps = (np.sqrt(n_s) - 1)/np.sqrt(n_s)
        else:
            eps = self.eps

        # Compute Kernel Matrix
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        K = pairwise.pairwise_kernels(Xs, Xs, metric=self.kernel,
                                      **kernel_params)
        K = (1/2) * (K + K.transpose())

        # Compute q
        kappa = pairwise.pairwise_kernels(Xs, Xt,
                                          metric=self.kernel,
                                          **kernel_params)
        kappa = (n_s/n_t) * np.dot(kappa, np.ones((n_t, 1)))
        
        P = matrix(K)
        q = -matrix(kappa)
        
        # Define constraints
        G = np.ones((2*n_s+2, n_s))
        G[1] = -G[1]
        G[2:n_s+2] = np.eye(n_s)
        G[n_s+2:n_s*2+2] = -np.eye(n_s)
        h = np.ones(2*n_s+2)
        h[0] = n_s*(1+eps)
        h[1] = n_s*(eps-1)
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
        return np.array(weights).ravel()

    
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
