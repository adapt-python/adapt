"""
Frustratingly Easy Domain Adaptation module.
"""

import warnings

import numpy as np
import tensorflow as tf

from adapt.utils import check_arrays, check_one_array, check_estimator


class FE:
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
        Estimator.
        
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
    >>> np.abs(model.predict(Xt, "src") - yt).mean()
    0.9846...
    >>> np.abs(model.predict(Xt, "tgt") - yt).mean()
    0.1010...

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.

    Note
    ----
    FE can be used for multi-source DA by giving list of source data
    for arguments Xs, ys of fit method : Xs = [Xs1, Xs2, ...],
    ys = [ys1, ys2, ...]
    """
    def __init__(self, estimator=None, copy=True, verbose=1, random_state=None):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.verbose = verbose
        self.estimator_ = check_estimator(estimator, copy=copy)
        self.copy = copy
        self.random_state = random_state


    def fit(self, Xs, ys, Xt, yt, **fit_params):
        """
        Fit FE.

        Parameters
        ----------
        Xs : numpy array or list of arrays
            Source input data. A list of
            arrays can be given for multi-source DA.

        ys : numpy array or list of arrays
            Source output data. A list of
            arrays can be given for multi-source DA.

        Xt : numpy array
            Target input data.
            
        yt : numpy array
            Target output data.

        fit_params : key, value arguments
            Arguments given to the fit method of the
            estimator.

        Returns
        -------
        self : returns an instance of self
        """
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt, True)        
        if self.verbose:
            print("Augmenting feature space...")
        Xs_emb, Xt_emb = self.fit_embeddings(Xs, Xt)
        if self.verbose:
            print("Fit estimator...")
        X = np.concatenate((Xs_emb, Xt_emb))
        y = np.concatenate((ys, yt))
        self.fit_estimator(X, y, **fit_params)
        return self

        
    def fit_embeddings(self, Xs, Xt):
        """
        Fit embeddings.
        
        Parameters
        ----------
        Xs : numpy array or list of arrays
            Source input data. A list of
            arrays can be given for multi-source DA.
            
        Xt : array
            Input target data.
            
        Returns
        -------
        Xs_emb, Xt_emb : embedded source and target data
        """
        Xs = check_one_array(Xs, multisource=True)           
        if self.verbose:
            print("Previous shape: %s"%str(Xs.shape))
        if isinstance(Xs, (tuple, list)):
            dim = Xs[0].shape[1]
            self.length_ = len(Xs)
            Xs_emb = np.concatenate(
                (Xs[0], np.zeros((len(Xs[0]), dim*self.length_)), Xs[0]),
                axis=-1)
            for i in range(1, self.length_):
                Xs_emb_i = np.concatenate((
                    np.zeros((len(Xs[i]), dim*i)),
                    Xs[i],
                    np.zeros((len(Xs[i]), dim*(self.length_-i))),
                    Xs[i]), axis=-1)
                Xs_emb = np.concatenate((Xs_emb, Xs_emb_i))
            Xt_emb = np.concatenate((np.zeros((len(Xt), dim*self.length_)),
                                     Xt, Xt), axis=-1)
        else:
            Xs_emb = np.concatenate((Xs, np.zeros(Xs.shape), Xs), axis=-1)
            Xt_emb = np.concatenate((np.zeros(Xt.shape), Xt, Xt), axis=-1)
        if self.verbose:
            print("New shape: %s"%str(Xs_emb.shape))
        return Xs_emb, Xt_emb
    
    
    def fit_estimator(self, X, y, **fit_params):
        """
        Fit estimator.
        
        Parameters
        ----------
        X : array
            Input data.
            
        y : array
            Output data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.
            
        Returns
        -------
        estimator_ : fitted estimator
        """
        X = check_one_array(X)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator_.fit(X, y, **fit_params)
        return self.estimator_


    def predict(self, X, domain="tgt"):
        """
        Return the predictions of ``estimator_`` on the
        augmented feature space.

        ``domain`` arguments specifies how features from ``X``
        will be considered: as ``"source"`` or
        ``"target"`` features.

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
        y_pred : array
            Prediction of ``estimator_``.

        Notes
        -----
        As FE is an anti-symetric feature-based method, one should indicates the
        domain of ``X`` in order to apply the appropriate feature transformation.
        """
        X = check_one_array(X)
        X_emb = self.predict_features(X, domain=domain)
        return self.estimator_.predict(X_emb)
    
    
    def predict_features(self, X, domain="tgt"):
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
        """
        X = check_one_array(X)
        if domain in ["tgt", "target"]:
            X_emb = np.concatenate((np.zeros(X.shape), X, X), axis=1)
        elif domain in ["src", "source"]:
            X_emb = np.concatenate((X, np.zeros(X.shape), X), axis=1)
        elif "src_" in domain:
            num_dom = int(domain.split("src_")[0])
            dim = X.shape[1]
            X_emb = np.concatenate((
                    np.zeros((len(X), dim*num_dom)),
                    X,
                    np.zeros((len(X), dim*(self.length_-num_dom))),
                    X), axis=-1)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        return X_emb
