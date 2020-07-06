import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression

# from tensorflow.keras import Sequential, Model
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.constraints import MinMaxNorm
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import multiply
# from tensorflow.keras import losses
# import tensorflow.keras.backend as K


class CORAL:
    """
    CORAL: CORrelation ALignment
    
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
        
    Cs_ : numpy array
        Correlation matrix of source features.
        
    Ct_ : numpy array
        Correlation matrix of target features.
        
    See also
    --------
    DeepCORAL

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1511.05547.pdf>`_ Sun B., Feng J., Saenko K. \
"Return of frustratingly easy domain adaptation". In AAAI, 2016.
    """
    def __init__(self, get_estimator=None, **kwargs):
        self.get_estimator = get_estimator
        self.kwargs = kwargs

        if self.get_estimator is None:
            self.get_estimator = LinearRegression


    def fit(self, X, y, src_index, tgt_index, tgt_index_labeled=None,
            sample_weight=None, **fit_params):
        """
        Fit estimator on the aligned feature space.

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

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)

        Xs = X[src_index]
        ys = y[src_index]
        Xt = X[tgt_index]
        yt = y[tgt_index]

        self.estimator_ = check_estimator(self.get_estimator, **self.kwargs)

        self.Cs_ = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1])
        self.Ct_ = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])

        Xs = np.matmul(Xs, linalg.inv(linalg.sqrtm(self.Cs_)))
        Xs = np.matmul(Xs, linalg.sqrtm(self.Ct_))
        
        if tgt_index_labeled is None:
            X = Xs
            y = ys
        else:
            X = np.concatenate((Xs, X[tgt_index_labeled]))
            y = np.concatenate((ys, y[tgt_index_labeled]))
        
        if sample_weight is None:
            self.estimator_.fit(X, y, **fit_params)
        else:
            if tgt_index_labeled is None:
                sample_weight = sample_weight[src_index]
            else:
                sample_weight = np.concatenate((
                    sample_weight[src_index],
                    sample_weight[tgt_index_labeled]
                ))
            self.estimator_.fit(X, y, sample_weight=sample_weight,
                                **fit_params)

        return self


    def predict(self, X):
        """
        Return the predictions of ``estimator_`` on ``X``.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Prediction of ``estimator_``.
        """
        return self.estimator_.predict(X)



# class DeepCORAL:
#     """
#     DeepCORAL: is a feature-based domain adaptation method.
    
#     Parameters
#     ----------
#     get_encoder: callable, optional
#         Constructor for encoder network.
#         The constructor should take at least
#         the "shape" argument.
        
#     get_task: callable, optional
#         Constructor for two networks: task and discriminer.
#         The constructor should take at least
#         the "shape" argument.
        
#     optimizer: tf Optimizer, optional (default="adam")
#         CORAL Optimizer
    
#     loss: tf.keras.losses object
#         If None mean_squared_error is used
        
#     kwargs: key, value arguments, optional
#         Additional arguments for constructors
#     """
#     def __init__(self, get_encoder, get_task, loss=losses.mean_squared_error,
#                  optimizer="adam", **kwargs):
#         self.loss = loss
#         self.optimizer = optimizer
#         self.get_encoder = get_encoder
#         self.get_task = get_task
#         self.kwargs = kwargs

    
#     def fit(self, X, y, index, **fit_params):
#         """
#         Fit CORAL
        
#         Parameters
#         ----------
#         X, y: numpy arrays
#             Input data
            
#         index: iterable
#             Index should contains 2 lists or 1D-arrays
#             corresponding to:
#             index[0]: indexes of source labeled data in X, y
#             index[1]: indexes of target unlabeled data in X, y
            
#         fit_params: key, value arguments
#             Arguments to pass to the fit method (epochs, batch_size...)
            
#         Returns
#         -------
#         self 
#         """
        
#         assert hasattr(index, "__iter__"), "index should be iterable"
#         assert len(index) == 2, "index length should be 2"
        
#         src_index = index[0]
#         tgt_index = index[1]
        
#         self._create_model(X.shape[1:], y.shape[1:])
        
#         max_size = max((len(src_index), len(tgt_index)))
#         resize_src_ind = np.array([src_index[i%len(src_index)]
#                                    for i in range(max_size)])
#         resize_tgt_ind = np.array([tgt_index[i%len(tgt_index)]
#                                    for i in range(max_size)])
        
#         self.model.fit([X[resize_src_ind], X[resize_tgt_ind], y[resize_src_ind]],
#                       **fit_params)
#         return self
    
    
#     def predict(self, X):
#         """
#         Predict method: return the prediction of task network
#         on the encoded features
        
#         Parameters
#         ----------
#         X: array
#             input data
            
#         Returns
#         -------
#         y_pred: array
#             prediction of task network
#         """
#         return self.task.predict(self.encoder.predict(X))
        
        
#     def _create_model(self, shape_X, shape_y):
        
#         self.encoder = self.get_encoder(shape=shape_X, **self.kwargs)
#         self.task = self.get_task(shape=self.encoder.output_shape[1:], **self.kwargs)
        
#         input_src = Input(shape_X)
#         input_tgt = Input(shape_X)
#         output_src = Input(shape_y)
        
#         encoded_src = self.encoder(input_src)
#         encoded_tgt = self.encoder(input_tgt)
        
#         tasked = self.task(encoded_task)
#         task_loss = self.loss(output_src, tasked)
        
#         n_s = encoded_src.shape[0]
#         ones_dot_encoded_src = K.dot(tf.ones((1, n_s)), encoded_src)
#         corr_src = (1 / (n_s - 1)) * (
#             K.transpose(encoded_src) * encoded_src -
#             (1 / n_s) * K.transpose(ones_dot_encoded_src) * ones_dot_encoded_src
#         )
#         n_t = encoded_tgt.shape[0]
#         ones_dot_encoded_tgt = K.dot(tf.ones((1, n_s)), encoded_tgt)
#         corr_tgt = (1 / (n_t - 1)) * (
#             K.transpose(encoded_tgt) * encoded_tgt -
#             (1 / n_t) * K.transpose(ones_dot_encoded_tgt) * ones_dot_encoded_tgt
#         )
        
#         corr_loss = K.mean(K.square(corr_src - corr_tgt))
        
#         loss = task_loss + corr_loss
        
#         self.model = Model([input_src, input_tgt, output_src],
#                            [encoded_src, encoded_tgt, tasked],
#                            name="CORAL")
#         self.model.add_loss(loss)
#         self.model.compile(optimizer=self.optimizer)
        
#         return self          
