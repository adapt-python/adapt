import numpy as np
from sklearn.base import check_array
from cvxopt import solvers, matrix

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.metrics import linear_discrepancy
from adapt.utils import set_random_seed


@make_insert_doc()
class LDM(BaseAdaptEstimator):
    """
    LDM : Linear Discrepancy Minimization
    
    LDM reweights the source instances in order to minimize
    the linear discrepancy between the reweighted source and
    the target data.
    
    The objective function is the following:
    
    .. math::
    
        \min_{||w||_1 = 1, w>0} \max_{||u||=1} |u^T M(w) u|
        
    Where:
    
    - :math:`M(w) = (1/n) X_T^T X_T - X^T_S diag(w) X_S`
    - :math:`X_S, X_T` are respectively the source dataset
      and the target dataset of size :math:`m` and :math:`n`
    
    Parameters
    ----------
    
    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
    
    estimator_ : object
        Estimator.
    
    See also
    --------
    KMM
    KLIEP
    
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import LDM
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = LDM(RidgeClassifier(), Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys)
    Fit weights...
    Initial Discrepancy : 0.328483
    Final Discrepancy : -0.000000
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.5
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0902.3430.pdf>`_ \
Y. Mansour, M. Mohri, and A. Rostamizadeh. "Domain \
adaptation: Learning bounds and algorithms". In COLT, 2009.
    """
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
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
        
        if self.verbose:
            disc = linear_discrepancy(Xs, Xt)
            print("Initial Discrepancy : %f"%disc)
        
        n = len(Xs)
        m = len(Xt)
        p = Xs.shape[1]

        Mis = []
        for i in range(n):
            Mis.append(Xs[i].reshape(-1,1).dot(Xs[i].reshape(1,-1)).ravel())

        M = np.stack(Mis, -1)
        lambda_I = np.eye(p).ravel().reshape(-1, 1)
        M0 = (1/m) * Xt.transpose().dot(Xt)
        M0 = M0.ravel().reshape(-1, 1)

        first_const_G = np.concatenate((-lambda_I, -M), axis=1)
        sec_const_G = np.concatenate((-lambda_I, M), axis=1)

        first_linear = np.ones((1, n+1))
        first_linear[0, 0] = 0.

        second_linear = -np.eye(n)
        second_linear = np.concatenate((np.zeros((n, 1)), second_linear), axis=1)

        G = matrix(np.concatenate((
            first_linear,
            -first_linear,
            second_linear,
            first_const_G,
            sec_const_G),
            axis=0)
        )

        h = matrix(np.concatenate((
            np.ones((1, 1)),
            -np.ones((1, 1)),
            np.zeros((n, 1)),
            -M0,
            M0)
        ))
        c = np.zeros((n+1, 1))
        c[0] = 1.
        c = matrix(c)

        dims = {'l': 2+n, 'q': [], 's':  [p, p]}

        sol = solvers.conelp(c, G, h, dims)
        
        if self.verbose:
            print("Final Discrepancy : %f"%sol['primal objective'])
        
        self.weights_ = np.array(sol["x"]).ravel()
        self.lambda_ = self.weights_[0]
        self.weights_ = np.clip(self.weights_[1:], 0., np.inf)
        return self.weights_
    
    
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