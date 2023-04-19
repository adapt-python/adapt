"""
Frustratingly Easy Domain Adaptation module.
"""

import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays


@make_insert_doc(supervised=True)
class FA(BaseAdaptEstimator):
    """
    FA: Feature Augmentation.

    FA consists in a feature augmentation method
    where each input feature vector is augmented as follow:

    - Source input feature vectors Xs are transformed into (Xs, **0**, Xs).
    - Target input feature vectors Xt are transformed into (**0**, Xt, Xt).

    Where **0** refers to the null vector of same size as Xs and Xt.

    The goal of this feature augmentation is 
    to separate features into the three following classes:

    - Specific source features (first part of the augmented vector)  which gives
      the specific behaviour on source domain.
    - Specific target features (second part of the augmented vector) which gives
      the specific behaviour on target domain.
    - General features (third part of the augmented vector) which have the
      same behaviour with respect to the task on both source and target domains.

    This feature-based method uses a few labeled target data and belongs to
    "supervised" domain adaptation methods.

    As FA consists only in a preprocessing step, any kind of estimator
    can be used to learn the task. This method handles both regression
    and classification tasks.

    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task.
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.

    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
        
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.
        
    n_domains_ : int
        Number of domains given in fit.
        
    See also
    --------
    CORAL
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import FA
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = FA(RidgeClassifier(), Xt=Xt[:10], yt=yt[:10], random_state=0)
    >>> model.fit(Xs, ys)
    Fit transform...
    Previous shape: (100, 2)
    New shape: (110, 6)
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.92

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.

    Notes
    -----
    FA can be used for multi-source DA by using the ``domains`` argument
    in the ``fit`` or ``fit_transform`` method. An example is given
    `[here] <https://github.com/adapt-python/adapt/issues/86>`_
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_transform(self, Xs, Xt, ys, yt, domains=None, **kwargs):
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
            
        domains : array (default=None)
            Vector giving the domain for each source
            data. Can be used for multisource purpose.
        
        kwargs : key, value argument
            Not used, present here for adapt consistency.
        
        Returns
        -------
        X_emb, y : embedded input and output data
        """
        if yt is None:
            raise ValueError("The target labels `yt` is `None`, FA is a supervised"
                             " domain adaptation method and need `yt` to be specified.")
        
        Xs, ys = check_arrays(Xs, ys)
        Xt, yt = check_arrays(Xt, yt)
        
        if self.verbose:
            print("Previous shape: %s"%str(Xs.shape))
        
        if domains is None:
            domains = np.zeros(len(Xs))
        
        domains = self._check_domains(domains).astype(int)
        
        self.n_domains_ = max(domains)+1
        dim = Xs.shape[-1]
        
        for i in range(self.n_domains_):
            Xs_emb_i = np.concatenate(
                (np.zeros((np.sum(domains==i), dim*i)),
                 Xs[domains==i],
                 np.zeros((np.sum(domains==i), dim*(self.n_domains_-i))),
                 Xs[domains==i]),
                axis=-1)
            if i == 0:
                Xs_emb = Xs_emb_i
                ys_emb = ys[domains==i]
            else:
                Xs_emb = np.concatenate((Xs_emb, Xs_emb_i))
                ys_emb = np.concatenate((ys_emb, ys[domains==i]))
        
        Xt_emb = np.concatenate((np.zeros((len(Xt), dim*self.n_domains_)),
                                 Xt, Xt), axis=-1)
        
        X = np.concatenate((Xs_emb, Xt_emb))
        y = np.concatenate((ys_emb, yt))
        
        if self.verbose:
            print("New shape: %s"%str(X.shape))
        return X, y


    def transform(self, X, domain="tgt"):
        """
        Return augmented features for X.
        
        In single source:
        
        - If ``domain="src"``, the method returns the array (X, **0**, X).
        - If ``domain="tgt"``, the method returns the array (**0**, X, X).
        
        With **0** the array of same shape as X with zeros everywhere.
        
        In single Multi-source:
        
        - If ``domain="src_%i"%i``, the method returns the array
          (X, [X]*i, **0**, [X]*(n_sources-i)).
        - If ``domain="tgt"``, the method returns the array (**0**, [X]*(n_sources+1)).
        
        Parameters
        ----------
        X : array
            Input data.

        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"`` feature augmentation,
            or "src_0", "src_1", ... in multisource setting.

        Returns
        -------
        X_emb : array
            Embeddings of X.

        Notes
        -----
        As FA is an anti-symetric feature-based method, one should indicates the
        domain of ``X`` in order to apply the appropriate feature transformation.
        """
        X = check_array(X, allow_nd=True)
        
        if not hasattr(self, "n_domains_"):
            raise NotFittedError("FA model is not fitted yet, please "
                                 "call 'fit_transform' or 'fit' first.")
        
        if domain in ["tgt", "target"]:
            X_emb = np.concatenate((np.zeros((len(X), X.shape[-1]*self.n_domains_)),
                                    X,
                                    X),
                                   axis=-1)
        elif domain in ["src", "source"]:
            X_emb = np.concatenate((X,
                                    np.zeros((len(X), X.shape[-1]*self.n_domains_)),
                                    X),
                                   axis=-1)
        elif "src_" in domain:
            num_dom = int(domain.split("src_")[1])
            dim = X.shape[-1]
            X_emb = np.concatenate((
                    np.zeros((len(X), dim*num_dom)),
                    X,
                    np.zeros((len(X), dim*(self.n_domains_-num_dom))),
                    X), axis=-1)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        return X_emb
