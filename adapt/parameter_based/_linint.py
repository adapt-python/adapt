"""
Frustratingly Easy Domain Adaptation module.
"""

import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays, set_random_seed


@make_insert_doc(supervised=True)
class LinInt(BaseAdaptEstimator):
    """
    LinInt: Linear Interpolation between SrcOnly and TgtOnly.
    
    LinInt linearly interpolates the predictions of the SrcOnly and
    TgtOnly models. The interpolation parameter is adjusted based on
    a small amount of target data removed from the training set
    of TgtOnly.

    Parameters
    ----------
    prop : float (default=0.5)
        Proportion between 0 and 1 of the data used
        to fit the TgtOnly model. The rest of the 
        target data are used to estimate the interpolation
        parameter.

    Attributes
    ----------
    estimator_src_ : object
        Fitted source estimator.
    
    estimator_ : object
        Fitted estimator.
          
    See also
    --------
    adapt.feature_based.FA
    adapt.feature_based.PRED
        
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from adapt.utils import make_regression_da
    >>> from adapt.parameter_based import LinInt
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> model = LinInt(Ridge(), Xt=Xt[:6], yt=yt[:6], prop=0.5,            
    ...              verbose=0, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.68...

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 prop=0.5,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit(self, Xs, ys, Xt=None, yt=None, **kwargs):
        """
        Fit LinInt.
        
        Parameters
        ----------
        Xs : array
            Source input data.
            
        ys : array
            Source output data.
            
        Xt : array
            Target input data.
            
        yt : array
            Target output data.
        
        kwargs : key, value argument
            Not used, present here for adapt consistency.
        
        Returns
        -------
        Xt_aug, yt : augmented input and output target data
        """        
        set_random_seed(self.random_state)
        
        Xs, ys = check_arrays(Xs, ys, accept_sparse=True)
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt, accept_sparse=True)
        
        shuffle_index = np.random.choice(len(Xt), len(Xt), replace=False)
        cut = int(len(Xt)*self.prop)
        Xt_train = Xt[shuffle_index[:cut]]
        Xt_test = Xt[shuffle_index[cut:]]
        yt_train = yt[shuffle_index[:cut]]
        yt_test = yt[shuffle_index[cut:]]
        
        self.estimator_src_ = self.fit_estimator(Xs, ys,
                                            warm_start=False,
                                            random_state=None)
        
        self.estimator_ = self.fit_estimator(Xt_train, yt_train,
                                            warm_start=False,
                                            random_state=None)
            
        self.interpolator_ = LinearRegression(fit_intercept=False)
        
        yp_src = self.estimator_src_.predict(Xt_test)
        yp_tgt = self.estimator_.predict(Xt_test)
        
        if len(yp_src.shape) < 2:
            yp_src = yp_src.reshape(-1, 1)
        if len(yp_tgt.shape) < 2:
            yp_tgt = yp_tgt.reshape(-1, 1)
        
        Xp = np.concatenate((yp_src, yp_tgt), axis=1)
        
        self.interpolator_.fit(Xp, yt_test)
        
        return self


    def predict(self, X):
        """
        Return LinInt predictions.
        
        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y : array
            Predictions
        """
        yp_src = self.estimator_src_.predict(X)
        yp_tgt = self.estimator_.predict(X)
        
        if len(yp_src.shape) < 2:
            yp_src = yp_src.reshape(-1, 1)
        if len(yp_tgt.shape) < 2:
            yp_tgt = yp_tgt.reshape(-1, 1)
        
        Xp = np.concatenate((yp_src, yp_tgt), axis=1)
        
        return self.interpolator_.predict(Xp)
    
    
    def score(self, X, y):
        """
        Compute R2 score
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        Returns
        -------
        score : float
            estimator score.
        """
        yp = self.predict(X)
        score = r2_score(y, yp)
        return score
