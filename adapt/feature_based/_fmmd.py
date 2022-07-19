import numpy as np
import tensorflow as tf
from sklearn.base import check_array
from cvxopt import solvers, matrix

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


def pairwise_X(X, Y):
    X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
    Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
    XY = tf.matmul(X, tf.transpose(Y))
    return X2 + tf.transpose(Y2) - 2*XY


def _get_optim_function(Xs, Xt, kernel="linear", gamma=1., degree=2, coef=1.):
    
    n = len(Xs)
    m = len(Xt)
    p = Xs.shape[1]
    
    Lxx = tf.ones((n,n), dtype=tf.float64) * (1./(n**2))
    Lxy = tf.ones((n,m), dtype=tf.float64) * (-1./(n*m))
    Lyy = tf.ones((m,m), dtype=tf.float64) * (1./(m**2))
    Lyx = tf.ones((m,n), dtype=tf.float64) * (-1./(n*m))

    L = tf.concat((Lxx, Lxy), axis=1)
    L = tf.concat((L, tf.concat((Lyx, Lyy), axis=1)), axis=0)
    
    if kernel == "linear":
        
        @tf.function
        def func(W):
            Kxx = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xs))
            Kyy = tf.matmul(tf.matmul(Xt, tf.linalg.diag(W**1)), tf.transpose(Xt))
            Kxy = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xt))

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((tf.transpose(Kxy), Kyy), axis=1)), axis=0)

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    elif kernel == "rbf":
        
        @tf.function
        def func(W):
            Kxx = pairwise_X(tf.matmul(Xs, tf.linalg.diag(W**1)), Xs)
            Kyy = pairwise_X(tf.matmul(Xt, tf.linalg.diag(W**1)), Xt)
            Kxy = pairwise_X(tf.matmul(Xs, tf.linalg.diag(W**1)), Xt)

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((tf.transpose(Kxy), Kyy), axis=1)), axis=0)
            K = tf.exp(-gamma * K)

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    elif kernel == "poly":
        
        @tf.function
        def func(W):
            Kxx = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xs))
            Kyy = tf.matmul(tf.matmul(Xt, tf.linalg.diag(W**1)), tf.transpose(Xt))
            Kxy = tf.matmul(tf.matmul(Xs, tf.linalg.diag(W**1)), tf.transpose(Xt))

            K = tf.concat((Kxx, Kxy), axis=1)
            K = tf.concat((K, tf.concat((tf.transpose(Kxy), Kyy), axis=1)), axis=0)
            K = (gamma * K + coef)**degree

            f = -tf.linalg.trace(tf.matmul(K, L))
            Df = tf.gradients(f, W)
            H = tf.hessians(f, W)
            return f, Df, H
        
    else:
        raise ValueError("kernel param should be in ['linear', 'rbf', 'poly']")
        
    return func
    

@make_insert_doc()
class fMMD(BaseAdaptEstimator):
    """
    fMMD : feature Selection with MMD
    
    LDM selects input features in order to minimize the
    maximum mean discrepancy (MMD) between the source and
    the target data.
    
    Parameters
    ----------
    threshold : float or 'auto' (default='auto')
        Threshold on ``features_scores_`` all
        feature with score above threshold will be
        removed.
        If 'auto' the threshold is chosen to maximize
        the difference between scores of selected features
        and removed ones.
        
    kernel : str (default='linear')
        Choose the kernel between
        ['linear', 'rbf', 'poly'].
        The kernels are computed as follows:
        
        - kernel = linear::
        
            k(X, Y) = <X, Y>
            
        - kernel = rbf::
        
            k(X, Y) = exp(gamma * ||X-Y||^2)
            
        - kernel = poly::
        
            poly(X, Y) = (gamma * <X, Y> + coef)^degree
        
    gamma : float (default=1.)
        Gamma multiplier for the 'rbf' and 
        'poly' kernels.
        
    degree : int (default=2)
        Degree of the 'poly' kernel
        
    coef : float (default=1.)
        Coef of the 'poly' kernel
    
    Attributes
    ----------
    estimator_ : object
        Estimator.
    
    selected_features_ : numpy array
        The selected features
        
    features_scores_ : numpy array
        The score attributed to each feature
    
    See also
    --------
    CORAL
    FA
    
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import fMMD
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = fMMD(RidgeClassifier(), Xt=Xt, kernel="linear", random_state=0, verbose=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.92
    
    References
    ----------
    .. [1] `[1] <https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication\
/Feature%20Selection%20for%20Transfer%20Learning.pdf>`_ S. Uguroglu and J. Carbonell. \
"Feature selection for transfer learning." In ECML PKDD. 2011.
    """

    def __init__(self,
                 estimator=None,
                 Xt=None,
                 threshold="auto",
                 kernel="linear",
                 gamma=1.,
                 degree=2,
                 coef=1.,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
    
    def fit_transform(self, Xs, Xt, **kwargs):
        """
        Fit embeddings.
        
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
        Xs_emb : embedded source data
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        n = len(Xs)
        m = len(Xt)
        p = Xs.shape[1]
        
        optim_func = _get_optim_function(tf.identity(Xs),
                                         tf.identity(Xt),
                                         self.kernel,
                                         self.gamma,
                                         self.degree,
                                         self.coef)
        
        def F(x=None, z=None):
            if x is None: return 0, matrix(1.0, (p,1))
            x = tf.identity(np.array(x).ravel())
            f, Df, H = optim_func(x)
            f = f.numpy()
            Df = Df[0].numpy().reshape(1, -1)
            H = H[0].numpy()
            if z is None: return matrix(f), matrix(Df)
            return matrix(f), matrix(Df), matrix(H)
        
        linear_const_G = -np.eye(p)
        squared_constraint_G = np.concatenate((np.zeros((1, p)), -np.eye(p)), axis=0)

        linear_const_h = np.zeros(p)
        squared_constraint_h = np.concatenate((np.ones(1), np.zeros(p)))

        G = matrix(np.concatenate((linear_const_G, squared_constraint_G)))
        h = matrix(np.concatenate((linear_const_h, squared_constraint_h)))
        dims = {'l': p, 'q': [p+1], 's':  []}
        
        solvers.options["show_progress"] = bool(self.verbose)
        
        sol = solvers.cp(F, G, h, dims)
        
        W = np.array(sol["x"]).ravel()
        
        self.selected_features_ = np.zeros(p, dtype=bool)
        
        if self.threshold == "auto":
            args = np.argsort(W).ravel()
            max_diff_arg = np.argmax(np.diff(W[args]))
            threshold = W[args[max_diff_arg]]
            self.selected_features_[W<=threshold] = 1
        else:
            self.selected_features_[W<=self.threshold] = 1
            
        if np.sum(self.selected_features_) == 0:
            raise Exception("No features selected")
            
        self.features_scores_ = W
        return Xs[:, self.selected_features_]
            
    
    def transform(self, X):
        """
        Return the projection of X on the selected featues.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        X_emb : array
            Embeddings of X.
        """
        X = check_array(X)
        return X[:, self.selected_features_]