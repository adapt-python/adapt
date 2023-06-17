"""
Regular Transfer
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse.linalg import lsqr
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

from adapt.base import BaseAdaptEstimator, BaseAdaptDeep, make_insert_doc
from adapt.utils import (check_arrays,
                         set_random_seed,
                         check_estimator,
                         check_network,
                         check_fitted_estimator,
                         get_default_task)


@make_insert_doc(supervised=True)
class RegularTransferLR(BaseAdaptEstimator):
    """
    Regular Transfer with Linear Regression
    
    RegularTransferLR is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting a linear estimator on target data
    according to an objective function regularized by the euclidean
    distance between source and target parameters:
    
    .. math::
    
        \\beta_T = \\underset{\\beta \in \\mathbb{R}^p}{\\text{argmin}}
        \\, ||X_T\\beta - y_T||^2 + \\lambda ||\\beta - \\beta_S||^2
        
    Where:
    
    - :math:`\\beta_T` are the target model parameters.
    - :math:`\\beta_S = \\underset{\\beta \\in \\mathbb{R}^p}{\\text{argmin}}
      \\, ||X_S\\beta - y_S||^2` are the source model parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`p` is the number of features in :math:`X_T`
      (:math:`+1` if ``intercept`` is True).
    - :math:`\\lambda` is a trade-off parameter.

    Parameters
    ----------        
    lambda_ : float (default=1.0)
        Trade-Off parameter.

    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
        
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from adapt.utils import make_regression_da
    >>> from adapt.parameter_based import RegularTransferLR
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> src_model = Ridge()
    >>> src_model.fit(Xs, ys)
    >>> print(src_model.score(Xt, yt))
    0.6771931378706197
    >>> tgt_model = RegularTransferLR(src_model, lambda_=1.)
    >>> tgt_model.fit(Xt[:3], yt[:3])
    >>> tgt_model.score(Xt, yt)
    0.6454964910964297
        
    See also
    --------
    RegularTransferLC, RegularTransferNN

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 lambda_=1.,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
                
        if not hasattr(estimator, "coef_"):
            raise ValueError("`estimator` argument has no ``coef_`` attribute, "
                             "please call `fit` on `estimator` or use "
                             "another estimator as `LinearRegression` or "
                             "`RidgeClassifier`.")
            
        estimator = check_fitted_estimator(estimator)

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)

    
    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit RegularTransferLR.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Not used. Here for sklearn compatibility.

        Returns
        -------
        self : returns an instance of self
        """        
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)

        if self.estimator_.fit_intercept:
            intercept_ = np.reshape(
                self.estimator_.intercept_,
                np.ones(self.estimator_.coef_.shape).mean(-1, keepdims=True).shape)
            beta_src = np.concatenate((
                intercept_,
                self.estimator_.coef_
            ), axis=-1)
            Xt = np.concatenate(
                (np.ones((len(Xt), 1)), Xt),
                axis=-1)
        else:
            beta_src = self.estimator_.coef_
        
        yt_ndim_below_one_ = False
        if yt.ndim <= 1:
            yt = yt.reshape(-1, 1)
            yt_ndim_below_one_ = True
        
        if beta_src.ndim <= 1:
            beta_src = beta_src.reshape(1, -1)
            
        if beta_src.shape[0] != yt.shape[1]:
            raise ValueError("The number of features of `yt`"
                             " does not match the number of coefs in 'estimator', "
                             "expected %i, got %i"%(beta_src.shape[0], yt.shape[1]))
        
        if beta_src.shape[1] != Xt.shape[1]:            
            beta_shape = beta_src.shape[1]; Xt_shape = Xt.shape[1]
            if self.estimator_.fit_intercept:
                beta_shape -= 1; Xt_shape -= 1
            raise ValueError("The number of features of `Xt`"
                             " does not match the number of coefs in 'estimator', "
                             "expected %i, got %i"%(beta_shape, Xt_shape))
            
        beta_tgt = []
        for i in range(yt.shape[1]):
            sol = lsqr(A=Xt, b=yt[:, i], damp=self.lambda_, x0=beta_src[i, :])
            beta_tgt.append(sol[0])
            
        beta_tgt = np.stack(beta_tgt, axis=0)
        
        if self.estimator_.fit_intercept:
            self.coef_ = beta_tgt[:, 1:]
            self.intercept_ = beta_tgt[:, 0]
        else:
            self.coef_ = beta_tgt
            
        if yt_ndim_below_one_:
            self.coef_ = self.coef_.reshape(-1)
            if self.estimator_.fit_intercept:
                self.intercept_ = self.intercept_[0]
            
        self.estimator_.coef_ = self.coef_
        if self.estimator_.fit_intercept:
            self.estimator_.intercept_ = self.intercept_
        return self

    

@make_insert_doc(supervised=True)
class RegularTransferLC(RegularTransferLR):
    """
    Regular Transfer for Linear Classification
    
    RegularTransferLC is a parameter-based domain adaptation method.
        
    This classifier first converts the target values into ``{-1, 1}``
    and then treats the problem as a regression task
    (multi-output regression in the multiclass case). It then fits
    the target data as a ``RegularTransferLR`` regressor, i.e it
    performs the following optimization:
    
    .. math::
    
        \\beta_T = \\underset{\\beta \in \\mathbb{R}^p}{\\text{argmin}}
        \\, ||X_T\\beta - y_T||^2 + \\lambda ||\\beta - \\beta_S||^2
        
    Where:
    
    - :math:`\\beta_T` are the target model parameters.
    - :math:`\\beta_S = \\underset{\\beta \\in \\mathbb{R}^p}{\\text{argmin}}
      \\, ||X_S\\beta - y_S||^2` are the source model parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`p` is the number of features in :math:`X_T`
      (:math:`+1` if ``intercept`` is True).
    - :math:`\\lambda` is a trade-off parameter.

    Parameters
    ----------        
    lambda_ : float (default=1.0)
        Trade-Off parameter.

    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
            
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.parameter_based import RegularTransferLC
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> src_model = RidgeClassifier()
    >>> src_model.fit(Xs, ys)
    >>> print(src_model.score(Xt, yt))
    0.88
    >>> tgt_model = RegularTransferLC(src_model, lambda_=10.)
    >>> tgt_model.fit(Xt[:3], yt[:3])
    >>> tgt_model.score(Xt, yt)
    0.92

    See also
    --------
    RegularTransferLR, RegularTransferNN

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    ### TODO reshape yt for multiclass.
    
    def fit(self, Xt=None, yt=None, **fit_params):       
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        
        _label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        _label_binarizer.fit(self.estimator.classes_)
        yt = _label_binarizer.transform(yt)
        
        print(yt.shape)
        
        return super().fit(Xt, yt, **fit_params)


@make_insert_doc(["task"], supervised=True)
class RegularTransferNN(BaseAdaptDeep):
    """
    Regular Transfer with Neural Network
    
    RegularTransferNN is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting a neural network on target data
    according to an objective function regularized by the euclidean
    distance between source and target parameters:
    
    .. math::
    
        \\beta_T = \\underset{\\beta=(\\beta_1, ... , \\beta_D)}{\\text{argmin}}
        \\, ||f(X_T, \\beta) - y_T||^2 + \sum_{i=1}^{D}
        \\lambda_i ||\\beta_i - {\\beta_S}_i||^2
        
    Where:
    
    - :math:`f` is a neural network with :math:`D` layers.
    - :math:`\\beta_T` are the parameters of the target neural network.
    - :math:`\\beta_S = \\underset{\\beta}{\\text{argmin}}
      \\, ||f(X_S,\\beta) - y_S||^2` are the source neural network parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`\\lambda_i` is the trade-off parameter of layer :math:`i`.
    
    Different trade-off can be given to the layer of the 
    neural network through the ``lambdas`` parameter.

    Parameters
    ----------        
    lambdas : float or list of float, optional (default=1.0)
        Trade-off parameters.
        If a list is given, values from ``lambdas`` are assigned
        successively to the list of ``network`` layers with 
        weights parameters going from the last layer to the first one.
        If the length of ``lambdas`` is smaller than the length of
        ``network`` layers list, the last trade-off value will be
        asigned to the remaining layers.

    Attributes
    ----------
    task_ : tensorflow Model
        Network.
        
    history_ : dict
        history of the losses and metrics across the epochs
        of the network training.
        
    Examples
    --------
    >>> from adapt.utils import make_regression_da
    >>> from adapt.parameter_based import RegularTransferNN
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> src_model = RegularTransferNN(loss="mse", lambdas=0., random_state=0)
    >>> src_model.fit(Xs, ys, epochs=100, verbose=0)
    >>> print(src_model.score(Xt, yt))
    1/1 [==============================] - 0s 127ms/step - loss: 0.2744
    0.27443504333496094
    >>> model = RegularTransferNN(src_model.task_, loss="mse", lambdas=1., random_state=0)
    >>> model.fit(Xt[:3], yt[:3], epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 109ms/step - loss: 0.0832
    0.08321201056241989
        
    See also
    --------
    RegularTransferLR, RegularTransferLC

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 task=None,
                 Xt=None,
                 yt=None,
                 lambdas=1.0,
                 regularizer="l2",
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        if not regularizer in ["l1", "l2"]:
            raise ValueError("`regularizer` argument should be "
                             "'l1' or 'l2', got, %s"%str(regularizer))
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit RegularTransferNN.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).

        Returns
        -------
        self : returns an instance of self
        """        
        Xt, yt = self._get_target_data(Xt, yt)
        Xs = Xt
        ys = yt
        return super().fit(Xs, ys, Xt=Xt, yt=yt, **fit_params)
    
    
    def _initialize_networks(self):
        if self.task is None:
            self.task_ = get_default_task(name="task")
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        self._add_regularization()
    
    
    def _get_regularizer(self, old_weight, weight, lambda_=1.):
        if self.regularizer == "l2":
            def regularizer():
                return lambda_ * tf.reduce_mean(tf.square(old_weight - weight))
        if self.regularizer == "l1":
            def regularizer():
                return lambda_ * tf.reduce_mean(tf.abs(old_weight - weight))
        return regularizer


    def _add_regularization(self):
        i = 0
        if not hasattr(self.lambdas, "__iter__"):
            lambdas = [self.lambdas]
        else:
            lambdas = self.lambdas
        
        for layer in reversed(self.task_.layers):
            if (hasattr(layer, "weights") and 
            layer.weights is not None and
            len(layer.weights) != 0):
                if i >= len(lambdas):
                    lambda_ = lambdas[-1]
                else:
                    lambda_ = lambdas[i]
                for weight in reversed(layer.weights):
                    old_weight = tf.identity(weight)
                    old_weight.trainable = False
                    self.add_loss(self._get_regularizer(
                        old_weight, weight, lambda_))
                i += 1
        
        
    def call(self, inputs):
        return self.task_(inputs)
    
    
    def transform(self, X):
        """
        Return X
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        return X
    
    
    def predict_disc(self, X):
        """
        Not used.
        """     
        pass
    
    
@make_insert_doc(supervised=True)
class RegularTransferGP(BaseAdaptEstimator):
    """
    Regular Transfer with Gaussian Process
    
    RegularTransferGP is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting the `alpha` coeficients of a
    Gaussian Process estimator on target data according to an
    objective function regularized by the euclidean distance between
    the source and target `alpha`:
    
    .. math::
    
        \\alpha_T = \\underset{\\alpha \in \\mathbb{R}^n}{\\text{argmin}}
        \\, ||K_{TS} \\alpha - y_T||^2 + \\lambda ||\\alpha - \\alpha_S||^2
        
    Where:
    
    - :math:`\\alpha_T` are the target model coeficients.
    - :math:`\\alpha_S = \\underset{\\alpha \\in \\mathbb{R}^n}{\\text{argmin}}
      \\, ||K_{SS} \\alpha - y_S||^2` are the source model coeficients.
    - :math:`y_S, y_T` are respectively the source and
      the target labels.
    - :math:`K_{SS}` is the pariwise kernel distance matrix between source
      input data.
    - :math:`K_{TS}` is the pariwise kernel distance matrix between target
      and source input data.
    - :math:`n` is the number of source data in :math:`X_S`
    - :math:`\\lambda` is a trade-off parameter. The larger :math:`\\lambda`
      the closer the target model will be from the source model.
    
    The ``estimator`` given to ``RegularTransferGP`` should be from classes
    ``sklearn.gaussian_process.GaussianProcessRegressor`` or 
    ``sklearn.gaussian_process.GaussianProcessClassifier``
    
    Parameters
    ----------
    lambda_ : float (default=1.0)
        Trade-Off parameter. For large ``lambda_``, the
        target model will be similar to the source model.

    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
        
    Examples
    --------
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    >>> from adapt.utils import make_regression_da
    >>> from adapt.parameter_based import RegularTransferGP
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> kernel = Matern() + WhiteKernel()
    >>> src_model = GaussianProcessRegressor(kernel)
    >>> src_model.fit(Xs, ys)
    >>> print(src_model.score(Xt, yt))
    -2.3409379221035382
    >>> tgt_model = RegularTransferGP(src_model, lambda_=1.)
    >>> tgt_model.fit(Xt[:3], yt[:3])
    >>> tgt_model.score(Xt, yt)
    -0.21947435769240653
        
    See also
    --------
    RegularTransferLR, RegularTransferNN

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    
    def __init__(self,
             estimator=None,
             Xt=None,
             yt=None,
             lambda_=1.,
             copy=True,
             verbose=1,
             random_state=None,
             **params):
                
        if not hasattr(estimator, "kernel_"):
            raise ValueError("`estimator` argument has no ``kernel_`` attribute, "
                             "please call `fit` on `estimator` or use "
                             "another estimator as `GaussianProcessRegressor` or "
                             "`GaussianProcessClassifier`.")
            
        estimator = check_fitted_estimator(estimator)

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)

    
    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit RegularTransferGP.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.

        fit_params : key, value arguments
            Not used. Here for sklearn compatibility.

        Returns
        -------
        self : returns an instance of self
        """
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
        
        if isinstance(self.estimator, GaussianProcessRegressor):
            src_linear_model = LinearRegression(fit_intercept=False)
            src_linear_model.coef_ = self.estimator_.alpha_.transpose()

            Kt = self.estimator_.kernel_(Xt, self.estimator_.X_train_)
            tgt_linear_model = RegularTransferLR(src_linear_model, lambda_=self.lambda_)
            
            tgt_linear_model.fit(Kt, yt)
            
            self.estimator_.alpha_ = np.copy(tgt_linear_model.coef_).transpose()
        
        elif isinstance(self.estimator, GaussianProcessClassifier):
            
            if hasattr(self.estimator_.base_estimator_, "estimators_"):
                for i in range(len(self.estimator_.base_estimator_.estimators_)):
                    c = self.estimator_.classes_[i]
                    if sum(yt == c) > 0:
                        yt_c = np.zeros(yt.shape[0])
                        yt_c[yt == c] = 1
                        self.estimator_.base_estimator_.estimators_[i] = self._fit_one_vs_one_classifier(
                        self.estimator_.base_estimator_.estimators_[i], Xt, yt_c)
                
            else:
                self.estimator_.base_estimator_ = self._fit_one_vs_one_classifier(
                self.estimator_.base_estimator_, Xt, yt)
        return self
    
    
    def _fit_one_vs_one_classifier(self, estimator, Xt, yt):
        src_linear_model = LinearRegression(fit_intercept=False)
        src_linear_model.coef_ = (estimator.y_train_ - estimator.pi_)
        src_linear_model.classes_ = estimator.classes_
        Kt = estimator.kernel_(Xt, estimator.X_train_)
        
        tgt_linear_model = RegularTransferLC(src_linear_model, lambda_=self.lambda_)

        tgt_linear_model.fit(Kt, yt)

        estimator.pi_ = (estimator.y_train_ - np.copy(tgt_linear_model.coef_).ravel())
        return estimator