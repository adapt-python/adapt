"""
Kernel Mean Matching
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise
from cvxopt import matrix, solvers

from adapt.utils import check_indexes, check_estimator


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
    
    - :math:`K_{ij} = k(x_i, x_j)` with :math:`x_i, x_j \\in X_S`
      and :math:k` a kernel.
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
    get_estimator : callable or object, optional (default=None)
        Constructor for the estimator.
        If a callable function is given it should return an estimator
        object (with ``fit`` and ``predict`` methods).
        If a class is given, a new instance of this class will
        be built and used as estimator.
        If get_estimator is ``None``, a ``LinearRegression`` object will be
        used by default as estimator.
        
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
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------  
    weights_ : numpy array
        Training instance weights.
    
    estimator_ : object
        Fitted estimator.

    See also
    --------
    KLIEP

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3075-correcting-sample-selection\
-bias-by-unlabeled-data.pdf>`_ J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf, \
and A. J. Smola. "Correcting sample selection bias by unlabeled data." In NIPS, 2007.
    """

    def __init__(self, get_estimator=None, B=1000, epsilon=None,
                 kernel="rbf", kernel_params=None, **kwargs):
        self.get_estimator = get_estimator
        self.B = B
        self.epsilon = epsilon
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.kwargs = kwargs
        
        if self.get_estimator is None:
            self.get_estimator = LinearRegression
        
        if self.kernel_params is None:
            self.kernel_params = {}


    def fit(self, X, y, src_index, tgt_index,
            tgt_index_labeled=None, **fit_params):
        """
        Fit KMM.

        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.
            
        tgt_index_labeled : iterable, optional (default=None)
            indexes of target labeled data in X, y.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)
        
        if tgt_index_labeled is None:
            Xs = X[src_index]
            ys = y[src_index]
        else:
            Xs = X[np.concatenate(
                (src_index, tgt_index_labeled)
            )]
            ys = y[np.concatenate(
                (src_index, tgt_index_labeled)
            )]
        Xt = X[tgt_index]
        
        n_s = len(Xs)
        n_t = len(Xt)
        
        # Get epsilon
        if self.epsilon is None:
            self.epsilon = (np.sqrt(n_s) - 1)/np.sqrt(n_s)

        # Compute Kernel Matrix
        K = pairwise.pairwise_kernels(Xs, Xs, metric=self.kernel,
                                      **self.kernel_params)
        K = (1/2) * (K + K.transpose())

        # Compute q
        kappa = pairwise.pairwise_kernels(Xs, Xt, metric=self.kernel,
                                          **self.kernel_params)
        q = -(n_s/n_t) * np.dot(kappa, np.ones((n_t, 1)))

        # Set up QP
        G = np.concatenate((np.ones((1,n_s)),
                            -np.ones((1,n_s)),
                            -np.eye(n_s),
                            np.eye(n_s)))
        h = np.concatenate((np.array([[n_s*(self.epsilon+1)]]),
                            np.array([[n_s*(self.epsilon-1)]]),
                            -np.zeros((n_s,1)),
                            np.ones((n_s,1))*self.B))
        P = matrix(K, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        
        # Solve QP
        solvers.options['show_progress'] = False
        weights = solvers.qp(P, q, G, h)['x']
        self.weights_ = np.array(weights).ravel()
        
        self.estimator_ = check_estimator(self.get_estimator, **self.kwargs)
        
        try:
            self.estimator_.fit(Xs, ys, 
                                sample_weight=self.weights_,
                                **fit_params)
        except:
            bootstrap_index = np.random.choice(
            len(Xs), size=len(Xs), replace=True,
            p=self.weights_ / self.weights_.sum())
            self.estimator_.fit(Xs[bootstrap_index], ys[bootstrap_index],
                          **fit_params)
        return self


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
        return self.estimator_.predict(X)
