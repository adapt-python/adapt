"""
Frustratingly Easy Domain Adaptation module.
"""

import warnings

import numpy as np
from sklearn.utils import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays


@make_insert_doc()
class FE(BaseAdaptEstimator):
    """
    FE: Frustratingly Easy Domain Adaptation.

    FE consists in a feature augmentation method
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

    As FE consists only in a preprocessing step, any kind of estimator
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
    mSDA
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import FE
    >>> np.random.seed(0)
    >>> Xs = 0.1 * np.random.randn(100, 1) + 1.
    >>> Xt = 0.1 * np.random.randn(100, 1) + 1.
    >>> ys = 0.1 * np.random.randn(100, 1) + 0.
    >>> yt = 0.1 * np.random.randn(100, 1) + 1.
    >>> model = FE()
    >>> model.fit(Xs, ys, Xt[:10], yt[:10]);
    Augmenting feature space...
    Previous shape: (100, 1)
    New shape: (100, 3)
    Fit estimator...
    >>> np.abs(model.predict(Xt, domain="src") - yt).mean()
    0.9846...
    >>> np.abs(model.predict(Xt, domain="tgt") - yt).mean()
    0.1010...

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.

    Notes
    -----
    FE can be used for multi-source DA by giving list of source data
    for arguments Xs, ys of fit method : Xs = [Xs1, Xs2, ...],
    ys = [ys1, ys2, ...]
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
        As FE is an anti-symetric feature-based method, one should indicates the
        domain of ``X`` in order to apply the appropriate feature transformation.
        """
        X = check_array(X, allow_nd=True)
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
