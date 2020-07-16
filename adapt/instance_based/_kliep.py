"""
Kullback-Leibler Importance Estimation Procedure
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise

from adapt.utils import check_indexes, check_estimator


class KLIEP:
    """
    KLIEP: Kullback–Leibler Importance Estimation Procedure
    
    KLIEP is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a the source instances **reweighting** which minimizes the 
    **Kullback-Leibler divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\alpha_i K_{\sigma}(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data.
    - :math:`\\alpha_i` are the basis functions coefficients.
    - :math:`K_{\sigma}(x, x_i) = \\text{exp}(-\\frac{||x - x_i||^2}{2\sigma^2})`
      are kernel functions of bandwidth :math:`\sigma`.
      
    KLIEP algorithm consists in finding the optimal :math:`\\alpha_i` according to
    the following optimization problem:
    
    .. math::
    
        \max_{\\alpha_i } \sum_{x_i \in X_T} \\text{log}(
        \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_i))
        
    Subject to:
    
    .. math::
    
        \sum_{x_k \in X_S} \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_k)) = n_S
        
    Where:
    
    - :math:`X_T` is the source input data of size :math:`n_S`.
    
    The above OP is solved through gradient descent algorithm.
    
    Furthemore a LCV procedure can be added to select the appropriate
    bandwidth :math:`\sigma`. The parameter is then selected using
    cross-validation on the :math:`J` score defined as follow:
    :math:`J = \\frac{1}{|\\mathcal{X}|} \\sum_{x \\in \\mathcal{X}} \\text{log}(w(x))`
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    KLIEP method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
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
        
    sigmas : float or list of float, optional (default=0.1)
        Kernel bandwidths.
        If ``sigmas`` is a list of multiple values, the
        kernel bandwidth is selected with the LCV procedure.
        
    cv : int, optional (default=5)
        Cross-validation split parameter.
        Used only if sigmas has more than one value.
        
    max_points : int, optional (default=100)
        Maximal number of target instances use to
        compute kernels.
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
        
    sigma_ = float
        Sigma selected for the kernel
        
    alphas_ = numpy array
        Basis functions coefficients.
        
    centers_ = numpy array
        Center points for kernels.
        
    self.j_scores_ = list of float
        List of J scores.
        
    estimator_ : object
        Fitted estimator.

    See also
    --------
    KMM

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3248-direct-importance-estimation\
-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf>`_ \
M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe. \
"Direct importance estimation with model selection and its application \
to covariateshift adaptation". In NIPS 2007
    """
    def __init__(self, get_estimator=None,
                 sigmas=None, cv=5, max_points=100, **kwargs):
        self.get_estimator = get_estimator
        self.sigmas = sigmas
        self.cv = cv
        self.max_points = max_points
        self.kwargs = kwargs
        
        if self.get_estimator is None:
            self.get_estimator = LinearRegression


    def fit(self, X, y, src_index, tgt_index,
            tgt_index_labeled=None, **fit_params):
        """
        Fit KLIEP.

        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.
            
        tgt_index_labeled : iterable, optional (default=None)
            indexes of target labeled data in X, y.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)
        
        if tgt_index_labeled is None:
            Xs = X[src_index]
            ys = y[src_index]
        else:
            Xs = X[np.concatenate(
                (src_index, tgt_index_labeled)
            )]
            ys = y[np.concatenate(
                (src_index, tgt_index_labeled)
            )]
        Xt = X[tgt_index]
        
        self.j_scores_ = []
        
        if hasattr(self.sigmas, "__len__") and len(self.sigmas) > 1:
            for sigma in self.sigmas:
                split = int(len(tgt_index) / self.cv)
                j_scores = []
                for i in range(self.cv):
                    if i == self.cv-1:
                        test_index = tgt_index[i * split:]
                    else:
                        test_index = tgt_index[i * split:
                                               (i + 1) * split]
                    train_index = np.array(
                        list(set(tgt_index) - set(test_index))
                    )

                    alphas, centers = self._fit(Xs,
                                                X[train_index],
                                                sigma)
                    
                    j_score = (1 / len(test_index)) * np.sum(np.log(
                        np.dot(
                            np.transpose(alphas),
                            pairwise.rbf_kernel(centers,
                                                X[test_index],
                                                sigma)
                        )
                    ))
                    j_scores.append(j_score)
                self.j_scores_.append(np.mean(j_score))
            self.sigma_ = self.sigmas[np.argmax(self.j_scores_)]
        else:
            try:
                self.sigma_ = self.sigmas[0]
            except:
                self.sigma_ = self.sigmas
        
        self.alphas_, self.centers_ = self._fit(Xs, Xt, self.sigma_)
        
        self.weights_ = np.dot(
            np.transpose(self.alphas_),
            pairwise.rbf_kernel(self.centers_, Xs, self.sigma_)
            ).ravel()
        
        self.estimator_ = check_estimator(self.get_estimator, **self.kwargs)
        
        try:
            self.estimator_.fit(Xs, ys, 
                                sample_weight=self.weights_,
                                **fit_params)
        except:
            bootstrap_index = np.random.choice(
            len(Xs), size=len(Xs), replace=True,
            p=self.weights_)
            self.estimator_.fit(Xs[bootstrap_index], ys[bootstrap_index],
                          **fit_params)
        return self


    def _fit(self, Xs, Xt, sigma):
        index_centers = np.random.choice(
                        len(Xt),
                        min(len(Xt), self.max_points),
                        replace=False)
        Xt_centers = Xt[index_centers]
        
        epsilon = 1e-4
        A = pairwise.rbf_kernel(Xt_centers, Xt, sigma)
        b = np.mean(pairwise.rbf_kernel(Xt_centers, Xs, sigma), axis=1)
        b = b.reshape(-1, 1)
        
        alpha = np.ones((len(Xt_centers), 1)) / len(Xt_centers)
        for k in range(5000):
            alpha += epsilon * np.dot(
                np.transpose(A), 1./np.dot(A, alpha)
            )
            alpha += b * ((((1-np.dot(np.transpose(b), alpha)) /
                            np.dot(np.transpose(b), b))))
            alpha = np.maximum(0, alpha)
            alpha /= np.dot(np.transpose(b), alpha)
        
        return alpha, Xt_centers


    def predict(self, X):
        """
        Return estimator predictions.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of estimator.
        """
        return self.estimator_.predict(X)
