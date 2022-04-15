from sklearn.decomposition import PCA
from sklearn.base import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc()
class SA(BaseAdaptEstimator):
    """
    SA : Subspace Alignment
    
    Linearly align the source domain to the target domain
    in a reduced PCA subspace of dimension ``n_components``.
    
    Parameters
    ----------
    n_components : int (default=None)
        Number of components of the PCA
        transformation. If ``None`` the
        number of components is equal
        to the input dimension of ``X``
    
    Attributes
    ----------
    estimator_ : object
        Fitted estimator.
        
    pca_src_ : sklearn PCA
        Source PCA
    
    pca_tgt_ : sklearn PCA
        Target PCA
        
    M_ : numpy array
        Alignment matrix
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import SA
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = SA(RidgeClassifier(), Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys)
    >>> model.score(Xt, yt)
    0.91
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/abs/1409.5241>`_ B. Fernando, A. Habrard, \
M. Sebban, and T. Tuytelaars. "Unsupervised visual domain adaptation using \
subspace alignment". In ICCV, 2013.
    """
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 n_components=None,
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
        
        self.pca_src_ = PCA(self.n_components)
        self.pca_tgt_ = PCA(self.n_components)
        
        self.pca_src_.fit(Xs)
        self.pca_tgt_.fit(Xt)
        
        self.M_  = self.pca_src_.components_.dot(
            self.pca_tgt_.components_.transpose())
        
        return self.pca_src_.transform(Xs).dot(self.M_)


    def transform(self, X, domain="tgt"):
        """
        Project X in the target subspace.
        
        The paramter ``domain`` specify if X should
        be considered as source or target data. As the
        transformation is assymetric, the source transformation
        should be applied on source data and the target
        transformation on target data.

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
            return self.pca_tgt_.transform(X)
        elif domain in ["src", "source"]:
            return self.pca_src_.transform(X).dot(self.M_)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)