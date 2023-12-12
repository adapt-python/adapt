"""
IWC
"""

import inspect

import numpy as np
from sklearn.utils import check_array
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays, set_random_seed, check_estimator

EPS = np.finfo(float).eps


@make_insert_doc()
class IWC(BaseAdaptEstimator):
    """
    IWC: Importance Weighting Classifier
    
    Importance weighting based on the output of a domain classifier
    which discriminates between source and target data.
    
    The source importance weighting are given with the following formula:
    
    .. math::
    
        w(x) = \\frac{1}{P(x \in Source)} - 1

    Parameters
    ----------
    classifier : object (default=None)
        Binary classifier trained to discriminate
        between source and target data.
        
    cl_params : dict (default=None)
        Dictionnary of parameters that will
        be given in the `fit` and/or `compile` methods
        of the classifier.

    Attributes
    ----------
    classifier_ : object
        Fitted classifier.
    
    estimator_ : object
        Fitted estimator.
          
    See also
    --------
    NearestNeighborsWeighting
    IWN
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import IWC
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = IWC(RidgeClassifier(0.), classifier=RidgeClassifier(0.),
    ...             Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys);
    >>> model.score(Xt, yt)
    0.74

    References
    ----------
    .. [1] `[1] <https://icml.cc/imls/conferences/2007/proceedings/papers/303.pdf>`_ \
Steffen Bickel, Michael Bruckner, Tobias Scheffer. "Discriminative Learning for Differing \
Training and Test Distributions". In ICML 2007
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 classifier=None,
                 cl_params=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_weights(self, Xs, Xt, warm_start=False, **kwargs):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        warm_start : bool (default=False)
            Weither to train the domain classifier
            from scratch or not.
            If False, the classifier is trained from scratch.
            
        kwargs : key, value argument
            Not used, present here for adapt consistency.
            
        Returns
        -------
        weights_ : sample weights
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        if self.cl_params is None:
            self.cl_params_ = {}
        else:
            self.cl_params_ = self.cl_params
        
        if (not warm_start) or (not hasattr(self, "classifier_")):
            if self.classifier is None:
                self.classifier_ = LogisticRegression(penalty="none")
            else:
                self.classifier_ = check_estimator(self.classifier,
                                                   copy=True,
                                                   force_copy=True)
            
        if hasattr(self.classifier_, "compile"):
            args = [
            p.name
            for p in inspect.signature(self.classifier_.compile).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            compile_params = {}
            for key, value in self.cl_params_.items():
                if key in args:
                    compile_params[key] = value
            self.classifier_.compile(**compile_params)
            
        args = [
            p.name
            for p in inspect.signature(self.classifier_.fit).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        fit_params = {}
        for key, value in self.cl_params_.items():
            if key in args:
                fit_params[key] = value
        
        X = np.concatenate((Xs, Xt))
        y = np.concatenate((np.ones(Xs.shape[0]), np.zeros(Xt.shape[0])))
        shuffle_index = np.random.choice(len(X), len(X), replace=False)
        X = X[shuffle_index]
        y = y[shuffle_index]
        
        self.classifier_.fit(X, y, **fit_params)
        
        if isinstance(self.classifier_, BaseEstimator):
            if hasattr(self.classifier_, "predict_proba"):
                y_pred = self.classifier_.predict_proba(Xs)[:, 1]
            elif hasattr(self.classifier_, "_predict_proba_lr"):
                y_pred = self.classifier_._predict_proba_lr(Xs)[:, 1]
            else:
                y_pred = self.classifier_.predict(Xs).ravel()
        else:
            y_pred = self.classifier_.predict(Xs).ravel()
        
        self.weights_ = 1. / np.clip(y_pred, EPS, 1.) - 1.
        
        return self.weights_
    
    
    def predict_weights(self, X=None):
        """
        Return fitted source weights
        
        If ``None``, the fitted source weights are returned.
        Else, sample weights are computing using the fitted
        ``classifier_``.
        
        Parameters
        ----------
        X : array (default=None)
            Input data.
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            if X is None:
                return self.weights_
            else:
                X = check_array(X)
                if isinstance(self.classifier_, BaseEstimator):
                    if hasattr(self.classifier_, "predict_proba"):
                        y_pred = self.classifier_.predict_proba(X)[:, 1]
                    elif hasattr(self.classifier_, "_predict_proba_lr"):
                        y_pred = self.classifier_._predict_proba_lr(X)[:, 1]
                    else:
                        y_pred = self.classifier_.predict(X).ravel()
                else:
                    y_pred = self.classifier_.predict(X).ravel()
                weights = 1. / (y_pred + EPS) - 1.
                return weights
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")