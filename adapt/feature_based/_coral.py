"""
Correlation Alignement Module.
"""

import numpy as np
from scipy import linalg
from sklearn.utils import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc()
class CORAL(BaseAdaptEstimator):
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
    
    - :math:`C_S = Cov(X_S) + \\lambda I_p`
    - :math:`C_T = Cov(X_T) + \\lambda I_p`
    - :math:`X_S = X_S C_S^{-\\frac{1}{2}}`
    - :math:`X_S^{enc} = X_S C_T^{\\frac{1}{2}}`
    
    Where :math:`\\lambda` is a regularization parameter.
    
    Notice that CORAL only uses labeled source and unlabeled target data.
    It belongs then to "unsupervised" domain adaptation methods.
    
    Parameters
    ----------
    lambda_ : float (default=1e-5)
        Regularization parameter. The larger
        ``lambda`` is, the less adaptation is performed.

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
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import CORAL
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = CORAL(RidgeClassifier(), Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys)
    Fit transform...
    Previous covariance difference: 0.013181
    New covariance difference: 0.000004
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.86
        
    See also
    --------
    DeepCORAL
    FA

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1511.05547.pdf>`_ Sun B., Feng J., Saenko K. \
"Return of frustratingly easy domain adaptation". In AAAI, 2016.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 lambda_=1e-5,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit_transform(self, Xs, Xt, **kwargs):
        """
        Fit embeddings.
        
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
        Xs_emb : embedded source data
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
        
        cov_Xs = np.cov(Xs, rowvar=False)
        cov_Xt = np.cov(Xt, rowvar=False)
        
        if self.verbose:
            print("Previous covariance difference: %f"%
                  (np.mean(np.abs(cov_Xs-cov_Xt))))
          
        self.Cs_ = cov_Xs + self.lambda_ * np.eye(Xs.shape[1])
        self.Ct_ = cov_Xt + self.lambda_ * np.eye(Xt.shape[1])
        
        Cs_sqrt_inv = linalg.inv(linalg.sqrtm(self.Cs_))
        Ct_sqrt = linalg.sqrtm(self.Ct_)
        
        if np.iscomplexobj(Cs_sqrt_inv):
            Cs_sqrt_inv = Cs_sqrt_inv.real
        if np.iscomplexobj(Ct_sqrt):
            Ct_sqrt = Ct_sqrt.real
        
        Xs_emb = np.matmul(Xs, Cs_sqrt_inv)
        Xs_emb = np.matmul(Xs_emb, Ct_sqrt)
        
        if self.verbose:
            new_cov_Xs = np.cov(Xs_emb, rowvar=False)
            print("New covariance difference: %f"%
                  (np.mean(np.abs(new_cov_Xs-cov_Xt))))
        return Xs_emb


    def transform(self, X, domain="tgt"):
        """
        Return aligned features for X.

        Parameters
        ----------
        X : array
            Input data.

        domain : str (default="tgt")
            Choose between ``"source", "src"`` or
            ``"target", "tgt"`` feature embedding.

        Returns
        -------
        X_emb : array
            Embeddings of X.
        """
        X = check_array(X)
        if domain in ["tgt", "target"]:
            X_emb = X
        elif domain in ["src", "source"]:
            Cs_sqrt_inv = linalg.inv(linalg.sqrtm(self.Cs_))
            Ct_sqrt = linalg.sqrtm(self.Ct_)

            if np.iscomplexobj(Cs_sqrt_inv):
                Cs_sqrt_inv = Cs_sqrt_inv.real
            if np.iscomplexobj(Ct_sqrt):
                Ct_sqrt = Ct_sqrt.real
            
            X_emb = np.matmul(X, Cs_sqrt_inv)
            X_emb = np.matmul(X_emb, Ct_sqrt)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        return X_emb
