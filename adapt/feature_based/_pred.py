"""
Frustratingly Easy Domain Adaptation module.
"""

import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays, check_estimator


@make_insert_doc(supervised=True)
class PRED(BaseAdaptEstimator):
    """
    PRED: Feature Augmentation with SrcOnly Prediction

    PRED uses the output of a source pretrain model as a feature in
    the target model. Specifically, PRED first trains a
    SrcOnly model. Then it runs the SrcOnly model on the target data.
    It uses the predictions made by the SrcOnly model as additional features
    and trains a second model on the target data, augmented with this new feature.

    Parameters
    ----------
    pretrain : bool (default=True)
        Weither to pretrain the estimator on
        source or not. If False, `estimator`
        should be already fitted.

    Attributes
    ----------
    estimator_src_ : object
        Fitted source estimator.
    
    estimator_ : object
        Fitted estimator.
          
    See also
    --------
    FA
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import PRED
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = PRED(RidgeClassifier(0.), Xt=Xt[[1, -1, -2]], yt=yt[[1, -1, -2]],
    ...              pretrain=True, verbose=0, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.77

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 copy=True,
                 pretrain=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_transform(self, Xs, Xt, ys, yt, **kwargs):
        """
        Fit embeddings.
        
        Parameters
        ----------
        Xs : array
            Source input data.
            
        Xt : array
            Target input data.
            
        ys : array
            Source output data.
            
        yt : array
            Target output data.
        
        kwargs : key, value argument
            Not used, present here for adapt consistency.
        
        Returns
        -------
        Xt_aug, yt : augmented input and output target data
        """        
        Xs, ys = check_arrays(Xs, ys)
        Xt, yt = check_arrays(Xt, yt)
        
        self.estimators_ = []
        
        if self.pretrain:
            estimator = self.fit_estimator(Xs, ys,
                                            warm_start=False,
                                            random_state=self.random_state)
            self.estimator_src_ = estimator
            del self.estimator_
        else:
            self.estimator_src_ = check_estimator(self.estimator,
                                              copy=self.copy,
                                              force_copy=True)
            
        yt_pred = self.estimator_src_.predict(Xt)
        
        if len(yt_pred.shape) < 2:
            yt_pred = yt_pred.reshape(-1, 1)
            
        X = np.concatenate((Xt, yt_pred), axis=-1)
        y = yt
        return X, y


    def transform(self, X, domain="tgt"):
        """
        Return augmented features for X.
        
        If `domain="tgt"`, the prediction of the source model on `X`
        are added to `X`.
        
        If `domain="src"`, `X` is returned.
        
        Parameters
        ----------
        X : array
            Input data.

        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"`` feature augmentation.

        Returns
        -------
        X_emb : array
            Embeddings of X.
        """
        X = check_array(X, allow_nd=True)
        
        if domain in ["tgt", "target"]:
            y_pred = self.estimator_src_.predict(X)
            if len(y_pred.shape) < 2:
                y_pred = y_pred.reshape(-1, 1)
            X_emb = np.concatenate((X, y_pred), axis=-1)
        elif domain in ["src", "source"]:
            X_emb = X
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        return X_emb
    
    
    def predict(self, X, domain=None, **predict_params):
        """
        Return estimator predictions after
        adaptation.
        
        If `domain="tgt"`, the input feature ``X`` are first transformed.
        Then the ``predict`` method of the fitted estimator
        ``estimator_`` is applied on the transformed ``X``.
        
        If `domain="src"`, ``estimator_src_`` is applied direclty
        on ``X``.
        
        Parameters
        ----------
        X : array
            input data
        
        domain : str (default=None)
            For antisymetric feature-based method,
            different transformation of the input X
            are applied for different domains. The domain
            should then be specified between "src" and "tgt".
            If ``None`` the default transformation is the
            target one.
        
        Returns
        -------
        y_pred : array
            prediction of the Adapt Model.
        """
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        if domain is None:
            domain = "tgt"
        X = self.transform(X, domain=domain)
        
        if domain in ["tgt", "target"]:
            return self.estimator_.predict(X, **predict_params)
        else:
            return self.estimator_src_.predict(X, **predict_params)


    def score(self, X, y, sample_weight=None, domain=None):
        """
        Return the estimator score.
        
        If `domain="tgt"`, the input feature ``X`` are first transformed.
        Then the ``score`` method of the fitted estimator
        ``estimator_`` is applied on the transformed ``X``.
        
        If `domain="src"`, ``estimator_src_`` is applied direclty
        on ``X``.
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        sample_weight : array (default=None)
            Sample weights
             
        domain : str (default=None)
            This parameter specifies for antisymetric
            feature-based method which transformation
            will be applied between "source" and "target".
            If ``None`` the transformation by default is
            the target one.
            
        Returns
        -------
        score : float
            estimator score.
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        
        if domain is None:
            domain = "tgt"
        X = self.transform(X, domain=domain)
        
        if domain in ["tgt", "target"]:
            estimator = self.estimator_
        else:
            estimator = self.estimator_src_
        
        if hasattr(estimator, "score"):
            score = estimator.score(X, y, sample_weight)
        elif hasattr(estimator, "evaluate"):
            if np.prod(X.shape) <= 10**8:
                score = estimator.evaluate(
                    X, y,
                    sample_weight=sample_weight,
                    batch_size=len(X)
                )
            else:
                score = estimator.evaluate(
                    X, y,
                    sample_weight=sample_weight
                )
            if isinstance(score, (tuple, list)):
                score = score[0]
        else:
            raise ValueError("Estimator does not implement"
                             " score or evaluate method")
        return score
