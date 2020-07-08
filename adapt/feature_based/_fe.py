"""
Frustratingly Easy Domain Adaptation module.
"""


import numpy as np
from sklearn.linear_model import LinearRegression

from ..utils import check_indexes, check_estimator


class FE:
    """
    FE: Frustratingly Easy Domain Adaptation.

    FE consists in a feature augmentation method
    where each input feature vector is augmented as follow:

    - Source input feature vectors Xs are transformed into (Xs, Xs, **0**).
    - Target input feature vectors Xt are transformed into (Xt, **0**, Xt).

    Where **0** refers to the null vector of same size as Xs and Xt.

    The goal of this feature augmentation is to help an estimator (given by
    ``get_estimator``) to separate features into the three following classes:

    - General features (first part of the augmented vector) which have the
      same behaviour with respect to the task on both source and target domains.
    - Specific source features (second part of the augmented vector)  which gives
      the specific behaviour on source domain.
    - Specific target features (third part of the augmented vector) which gives
      the specific behaviour on target domain.

    This feature-based method uses a few labeled target data and belongs to
    "supervised" domain adaptation methods.

    As FE consists only in a preprocessing step, any kind of estimator
    can be used to learn the task. This method handles both regression
    and classification tasks.

    Parameters
    ----------
    get_estimator : callable or object, optional (default=None)
        Constructor for the estimator.
        If a callable function is given it should return an estimator
        object (with ``fit`` and ``predict`` methods).
        If a class is given, a new instance of this class will
        be built and used as estimator.
        If get_estimator is ``None``, a ``LinearRegression`` object will be
        used by default as estimator.

    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/0907.1815\
.pdf>`_ Daume III, H. "Frustratingly easy domain adaptation". In ACL, 2007.
    """
    def __init__(self, get_estimator=None, **kwargs):
        self.get_estimator = get_estimator
        self.kwargs = kwargs

        if self.get_estimator is None:
            self.get_estimator = LinearRegression


    def fit(self, X, y, src_index, tgt_index, sample_weight=None, **fit_params):
        """
        Fit estimator on the augmented feature space.

        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target labeled data in X, y.

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index)

        Xs = X[src_index]
        ys = y[src_index]
        Xt = X[tgt_index]
        yt = y[tgt_index]

        self.estimator_ = check_estimator(self.get_estimator, **self.kwargs)

        Xs = np.concatenate((Xs, np.zeros(Xs.shape), Xs), axis=-1)
        Xt = np.concatenate((np.zeros(Xt.shape), Xt, Xt), axis=-1)

        X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))

        if sample_weight is None:
            self.estimator_.fit(X, y, **fit_params)
        else:
            sample_weight = np.concatenate((
                sample_weight[src_index],
                sample_weight[tgt_index]
            ))
            self.estimator_.fit(X, y, sample_weight=sample_weight,
                                **fit_params)

        return self


    def predict(self, X, domain="target"):
        """
        Return the predictions of ``estimator_`` on the augmented feature space.

        ``domain`` arguments specify how features from ``X`` will be considered:
        as ``"source"`` or ``"target"`` features.

        Parameters
        ----------
        X : array
            Input data.

        domain : str, optional (default="target")
            Choose between ``"source"`` and ``"target"`` pre-processing.

        Returns
        -------
        y_pred : array
            Prediction of ``estimator_``.

        Notes
        -----
        As FE is an anti-symetric feature-based method, one should indicates the
        domain of ``X`` in order to apply the appropriate feature transformation.
        """
        if domain == "target":
            X = np.concatenate((np.zeros(X.shape), X, X), axis=1)
        elif domain == "source":
            X = np.concatenate((X, np.zeros(X.shape), X), axis=1)
        else:
            raise ValueError("Choose between source or target for domain name")
        return self.estimator_.predict(X)
