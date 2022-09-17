import numpy as np
from sklearn.base import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc(supervised=True)
class BalancedWeighting(BaseAdaptEstimator):
    """
    BW : Balanced Weighting
    
    Fit the estimator :math:`h` on source and target labeled data
    according to the modified loss:
    
    .. math::
    
        \min_{h} (1-\gamma) \mathcal{L}(h(X_S), y_S) + \gamma \mathcal{L}(h(X_T), y_T)
        
    Where:
    
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the labeled source
      and target data.
    - :math:`\mathcal{L}` is the estimator loss
    - :math:`\gamma` is the ratio parameter
    
    Parameters
    ----------
    gamma : float (default=0.5)
        ratio between 0 and 1 correspond to the importance
        given to the target labeled data. When `ratio=1`, the
        estimator is only fitted on target data. `ratio=0.5`
        corresponds to a balanced training.
    
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
    >>> from adapt.instance_based import BalancedWeighting
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = BalancedWeighting(RidgeClassifier(), gamma=0.5, Xt=Xt[:3], yt=yt[:3],
    ...                           verbose=0, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.93
    
    See also
    --------
    TrAdaBoost
    TrAdaBoostR2
    WANN
    
    References
    ----------
    .. [1] `[1] <https://openreview.net/forum?id=SybwYsbdWH>`_ P. Wu, T. G. Dietterich. \
"Improving SVM accuracy by training on auxiliary data sources". In ICML 2004
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 gamma=0.5,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
        
    def fit_weights(self, Xs, Xt, ys, yt, **kwargs):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        ys : array
            Source labels.
            
        yt : array
            Target labels.
            
        kwargs : key, value argument
            Not used, present here for adapt consistency.
            
        Returns
        -------
        weights_ : sample weights
        
        X : concatenation of Xs and Xt
        
        y : concatenation of ys and yt
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))
        
        src_weights = np.ones(Xs.shape[0]) * Xt.shape[0] * (1-self.gamma)
        tgt_weights = np.ones(Xt.shape[0]) * Xs.shape[0] * self.gamma
        
        self.weights_ = np.concatenate((src_weights, tgt_weights))
        self.weights_ /= np.mean(self.weights_)
        
        return self.weights_, X, y
    
    
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