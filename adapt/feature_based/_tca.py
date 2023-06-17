"""
TCA
"""

import numpy as np
from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
from scipy import linalg

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc()
class TCA(BaseAdaptEstimator):
    """
    TCA :  Transfer Component Analysis
    
    Parameters
    ----------
    n_components : int or float (default=None)
        Number of components to keep.
    
    mu : float (default=0.1)
        Regularization parameter. The larger
        ``mu`` is, the less adaptation is performed.

    Attributes
    ----------
    estimator_ : object
        Estimator.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import TCA
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = TCA(RidgeClassifier(), Xt=Xt, n_components=1, mu=0.1, 
    ...             kernel="rbf", gamma=0.1, verbose=0, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.93
        
    See also
    --------
    CORAL
    FA
    
    References
    ----------
    .. [1] `[1] <https://www.cse.ust.hk/~qyang/Docs/2009/TCA.pdf>`_ S. J. Pan, \
I. W. Tsang, J. T. Kwok and Q. Yang. "Domain Adaptation via Transfer Component \
Analysis". In IEEE transactions on neural networks 2010
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 n_components=20,
                 mu=0.1,
                 kernel="rbf",
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
        
        self.Xs_ = Xs
        self.Xt_ = Xt
        
        n = len(Xs)
        m = len(Xt)
        
        # Compute Kernel Matrix K
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        
        Kss = pairwise.pairwise_kernels(Xs, Xs, metric=self.kernel, **kernel_params)
        Ktt = pairwise.pairwise_kernels(Xt, Xt, metric=self.kernel, **kernel_params)
        Kst = pairwise.pairwise_kernels(Xs, Xt, metric=self.kernel, **kernel_params)

        K = np.concatenate((Kss, Kst), axis=1)
        K = np.concatenate((K, np.concatenate((Kst.transpose(), Ktt), axis=1)), axis=0)
        
        # Compute L
        Lss = np.ones((n,n)) * (1./(n**2))
        Ltt = np.ones((m,m)) * (1./(m**2))
        Lst = np.ones((n,m)) * (-1./(n*m))

        L = np.concatenate((Lss, Lst), axis=1)
        L = np.concatenate((L, np.concatenate((Lst.transpose(), Ltt), axis=1)), axis=0)
        
        # Compute H
        H = np.eye(n+m) - 1/(n+m) * np.ones((n+m, n+m))
        
        # Compute solution
        a = np.eye(n+m) + self.mu * K.dot(L.dot(K))
        b = K.dot(H.dot(K))
        sol = linalg.lstsq(a, b)[0]
        
        values, vectors = linalg.eigh(sol)
        
        args = np.argsort(np.abs(values))[::-1][:self.n_components]

        self.vectors_ = np.real(vectors[:, args])

        Xs_enc = K.dot(self.vectors_)[:n]

        return Xs_enc


    def transform(self, X, domain="tgt"):
        """
        Return aligned features for X.

        Parameters
        ----------
        X : array
            Input data.

        domain : str (default="tgt")
            Choose between ``"source", "src"`` or
            ``"target", "tgt"`` feature embedding.

        Returns
        -------
        X_emb : array
            Embeddings of X.
        """
        X = check_array(X)
        
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        
        Kss = pairwise.pairwise_kernels(X, self.Xs_, metric=self.kernel, **kernel_params)
        Kst = pairwise.pairwise_kernels(X, self.Xt_, metric=self.kernel, **kernel_params)

        K = np.concatenate((Kss, Kst), axis=1)
        
        return K.dot(self.vectors_)