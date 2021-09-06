"""
Correlation Alignement Module.
"""

import copy
import warnings

import numpy as np
from scipy import linalg
import tensorflow as tf

from adapt.utils import check_arrays, check_one_array, check_estimator


class CORAL:
    """
    CORAL: CORrelation ALignment
    
    CORAL is a feature based domain adaptation method which minimizes
    domain shift by aligning the second-order statistics of source and
    target distributions.
    
    The method transforms source features in order to minimize the
    Frobenius norm between the correlation matrix of the input target
    data and the one of the transformed input source data.
    
    The source features transformation is described by the following
    optimization problem:
    
    .. math::
        
        \min_{A}{||A^T C_S A - C_T||_F^2}
        
    Where:
    
    - :math:`A` is the feature transformation matrix such that
    :math:`X_S^{enc} = X_S A`
    - :math:`C_S` is the correlation matrix of input source data
    - :math:`C_T` is the correlation matrix of input target data
    
    The solution of this OP can be written with an explicit formula
    and the features transformation can be computed through this
    four steps algorithm:
    
    - :math:`C_S = \\lambda Cov(X_S) + I_p`
    - :math:`C_T = \\lambda Cov(X_T) + I_p`
    - :math:`X_S = X_S C_S^{-\\frac{1}{2}}`
    - :math:`X_S^{enc} = X_S C_T^{\\frac{1}{2}}`
    
    Where :math:`\\lambda` is a regularization parameter.
    
    Notice that CORAL only uses labeled source and unlabeled target data.
    It belongs then to "unsupervised" domain adaptation methods.
    
    Parameters
    ----------
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task. 
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.

    lambda_ : float (default=1.)
        Trade-off parameter. If ``lambda_`` is null,
        then no feature transformation is performed.
        
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
        
    Cs_ : numpy array
        Correlation matrix of source features.
        
    Ct_ : numpy array
        Correlation matrix of target features.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import CORAL
    >>> Xs = np.random.multivariate_normal(
    ...      np.array([0, 0]), np.array([[0.001, 0], [0, 1]]), 100)
    >>> Xt = np.random.multivariate_normal(
    ...      np.array([0, 0]), np.array([[0.1, 0.2], [0.2, 0.5]]), 100)
    >>> ys = np.zeros(100)
    >>> yt = np.zeros(100)
    >>> ys[Xs[:, 1]>0] = 1
    >>> yt[(Xt[:, 1]-0.5*Xt[:, 0])>0] = 1
    >>> model = CORAL(lambda_=0.)
    >>> model.fit(Xs, ys, Xt);
    Covariance Matrix alignement...
    Previous covariance difference: 0.258273
    New covariance difference: 0.258273
    Fit estimator...
    >>> model.estimator_.score(Xt, yt)
    0.5750...
    >>> model = CORAL(lambda_=100.)
    >>> model.fit(Xs, ys, Xt);
    Covariance Matrix alignement...
    Previous covariance difference: 0.258273
    New covariance difference: 0.040564
    Fit estimator...
    >>> model.estimator_.score(Xt, yt)
    0.5992...
        
    See also
    --------
    DeepCORAL
    FE
    mSDA

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1511.05547.pdf>`_ Sun B., Feng J., Saenko K. \
"Return of frustratingly easy domain adaptation". In AAAI, 2016.
    """
    def __init__(self, estimator=None, lambda_=1.,
                 copy=True, verbose=1, random_state=None):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.estimator_ = check_estimator(estimator, copy=copy)
        self.lambda_ = lambda_
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state


    def fit(self, Xs, ys, Xt, **fit_params):
        """
        Perfrom correlation alignement on input source data to match 
        input target data.
        Then fit estimator on the aligned source data and the labeled
        target ones.

        Parameters
        ----------
        Xs : numpy array
            Source input data.

        ys : numpy array
            Source output data.

        Xt : numpy array
            Target input data.

        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.

        Returns
        -------
        self : returns an instance of self
        """
        Xs, ys, Xt, _ = check_arrays(Xs, ys, Xt, None)
        if self.verbose:
            print("Covariance Matrix alignement...")
        Xs_emb, Xt = self.fit_embeddings(Xs, Xt)
        if self.verbose:
            print("Fit estimator...")
        self.fit_estimator(Xs_emb, ys, **fit_params)
        return self
        
    
    def fit_embeddings(self, Xs, Xt):
        """
        Fit embeddings.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        Returns
        -------
        Xs_emb, Xt_emb : embedded source and target data
        """
        cov_Xs = np.cov(Xs, rowvar=False)
        cov_Xt = np.cov(Xt, rowvar=False)
        
        if self.verbose:
            print("Previous covariance difference: %f"%
                  (np.mean(np.abs(cov_Xs-cov_Xt))))
          
        self.Cs_ = self.lambda_ * cov_Xs + np.eye(Xs.shape[1])
        self.Ct_ = self.lambda_ * cov_Xt + np.eye(Xt.shape[1])
        Xs_emb = np.matmul(Xs, linalg.inv(linalg.sqrtm(self.Cs_)))
        Xs_emb = np.matmul(Xs_emb, linalg.sqrtm(self.Ct_))
        
        if self.verbose:
            new_cov_Xs = np.cov(Xs_emb, rowvar=False)
            print("New covariance difference: %f"%
                  (np.mean(np.abs(new_cov_Xs-cov_Xt))))
        return Xs_emb, Xt
    
    
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
        Return the predictions of the estimator.

        Parameters
        ----------
        X : array
            Input data.
            
        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"`` feature embedding.

        Returns
        -------
        y_pred : array
            Prediction of ``estimator_``.
        """
        X = check_one_array(X)
        return self.estimator_.predict(
            self.predict_features(X, domain))


    def predict_features(self, X, domain="tgt"):
        """
        Return aligned features for X.

        Parameters
        ----------
        X : array
            Input data.

        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"`` feature embedding.

        Returns
        -------
        X_emb : array
            Embeddings of X.
        """
        X = check_one_array(X)
        if domain in ["tgt", "target"]:
            X_emb = X
        elif domain in ["src", "source"]:
            X_emb = np.matmul(X, linalg.inv(linalg.sqrtm(self.Cs_)))
            X_emb = np.matmul(X_emb, linalg.sqrtm(self.Ct_))
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        return X_emb
