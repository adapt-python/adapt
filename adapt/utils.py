"""
Utility functions for adapt package.
"""

import warnings
import inspect
from copy import deepcopy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
try:
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
except:
    from scikeras.wrappers import KerasClassifier, KerasRegressor
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Input
from tensorflow.keras.models import clone_model
from tensorflow.keras.initializers import GlorotUniform


class UpdateLambda(tf.keras.callbacks.Callback):
    """
    Update Lambda trade-off

    This Callback increases the ``lambda_`` trade-off parameter
    at each batch.

    The trade-off is increased from ``lambda_init`` to ``lambda_max``
    in ``max_steps`` number of gradient steps according to the
    following formula:

    ``lambda_`` = A * [ 2/(1 + exp(-``gamma`` * p)) - 1. ] + B
    
    With p increasing from 0 to 1 and A, B two constants.
    
    Parameters
    ----------
    lambda_init : float (default=0.)
        Initial trade-off
        
    lambda_max : float (default=1.)
        Trade-off after ``max_steps`` gradient updates.
        
    max_steps : int (default=1000)
        Number of gradient updates before getting ``lambda_max``
        
    gamma : float (default=1.)
        Speed factor. High ``gamma`` will increase the speed of
        ``lambda_`` increment.
    """
    def __init__(self, lambda_init=0., lambda_max=1., max_steps=1000, gamma=1.):
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.max_steps = max_steps
        self.gamma = gamma
        self.steps = 0.

    def on_batch_end(self, batch, logs=None):
        self.steps += 1.
        progress = min(self.steps / self.max_steps, 1.)
        lambda_ = 2. / (1. + tf.exp(-self.gamma * progress)) - 1.
        lambda_ /= (2 / (1. + tf.exp(-self.gamma)) - 1.)
        lambda_ *= (self.lambda_max - self.lambda_init)
        lambda_ += self.lambda_init
        self.model.lambda_.assign(lambda_)


def accuracy(y_true, y_pred):
    """
    Custom accuracy function which can handle
    probas vector in both binary and multi classification
    
    Parameters
    ----------
    y_true : Tensor
        True tensor.
        
    y_pred : Tensor
        Predicted tensor.
        
    Returns
    -------
    Boolean Tensor
    """
    # TODO: accuracy can't handle 1D ys.
    multi_columns_t = K.cast(K.greater(K.shape(y_true)[1], 1),
                           "float32")
    binary_t = K.reshape(K.sum(K.cast(K.greater(y_true, 0.5),
                                    "float32"), axis=-1), (-1,))
    multi_t = K.reshape(K.cast(K.argmax(y_true, axis=-1),
                             "float32"), (-1,))
    y_true = ((1 - multi_columns_t) * binary_t +
              multi_columns_t * multi_t)
    
    multi_columns_p = K.cast(K.greater(K.shape(y_pred)[1], 1),
                           "float32")
    binary_p = K.reshape(K.sum(K.cast(K.greater(y_pred, 0.5),
                                    "float32"), axis=-1), (-1,))
    multi_p = K.reshape(K.cast(K.argmax(y_pred, axis=-1),
                             "float32"), (-1,))
    y_pred = ((1 - multi_columns_p) * binary_p +
              multi_columns_p * multi_p)        
    return tf.keras.metrics.get("acc")(y_true, y_pred)


def predict(self, x, **kwargs):
    if hasattr(x, "shape") and (np.prod(x.shape) <= 10**8):
        pred = self.__call__(tf.identity(x)).numpy()
    else:
        pred = Sequential.predict(self, x, **kwargs)
    return pred

      
def check_arrays(X, y, **kwargs):
    """
    Check arrays and reshape 1D array in 2D array
    of shape (-1, 1). Check if the length of X
    match the length of y.

    Parameters
    ----------
    X : numpy array
        Input data.

    y : numpy array
        Output data.
        
    Returns
    -------
    X, y
    """    
    X = check_array(X, ensure_2d=True, allow_nd=True, **kwargs)
    y = check_array(y, ensure_2d=False, allow_nd=True, dtype=None, **kwargs)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Length of X and y mismatch: %i != %i"%
                         (X.shape[0],y.shape[0]))
    return X, y


def check_estimator(estimator=None, copy=True,
                    name=None,
                    display_name="estimator",
                    task=None,
                    force_copy=False):
    """
    Check estimator.

    Check that ``estimator`` is a sklearn ``BaseEstimator``
    or a tensorflow ``Model``.

    Parameters
    ----------
    estimator : sklearn BaseEstimator or tensorflow Model
        Estimator. If ``None`` a LinearRegression instance
        or a LogisticRegression instance is returned
        depending on the ``task`` argument.

    copy : boolean (default=False)
        Whether to return a copy of the estimator or not.
        If cloning fail, a warning is raised.

    display_name: str (default="estimator")
        Name to display if an error or warning is raised

    task : str (default=None)
        Task at hand. Possible value : 
        (``None``, ``"reg"``, ``"class"``)

    force_copy : boolean (default=False)
        If True, an error is raised if the cloning failed.
    """
    if estimator is None:
        if task == "class":
            estimator = LogisticRegression()
        else:
            estimator = LinearRegression()
    
    # TODO, add KerasWrappers in doc and error message
    if isinstance(estimator, (BaseEstimator, KerasClassifier, KerasRegressor)):
        if (isinstance(estimator, ClassifierMixin) and task=="reg"):
            raise ValueError("`%s` argument is a sklearn `ClassifierMixin` instance "
                             "whereas the considered object handles only regression task. "
                             "Please provide a sklearn `RegressionMixin` instance or a "
                             "tensorflow Model instance."%display_name)
        if (isinstance(estimator, RegressorMixin) and task=="class"):
            raise ValueError("`%s` argument is a sklearn `RegressionMixin` instance "
                             "whereas the considered object handles only classification task. "
                             "Please provide a sklearn `ClassifierMixin` instance or a "
                             "tensorflow Model instance."%display_name)
        if copy:
            try:
                if isinstance(estimator, (KerasClassifier, KerasRegressor)):
                    # TODO, copy fitted parameters and Model
                    new_estimator = clone(estimator)
                else:
                    new_estimator = deepcopy(estimator)
            except Exception as e:
                if force_copy:
                    raise ValueError("`%s` argument can't be duplicated. "
                                     "Recorded exception: %s. "%
                                     (display_name, e))
                else:
                    warnings.warn("`%s` argument can't be duplicated. "
                                  "Recorded exception: %s. "
                                  "The current estimator will be used. "
                                  "Use `copy=False` to hide this warning."%
                                  (display_name, e))
                    new_estimator = estimator
        else:
            new_estimator = estimator    
    elif isinstance(estimator, Model):
        new_estimator = check_network(network=estimator,
                                    copy=copy, 
                                  name=name,
                                  force_copy=force_copy)
    else:
        raise ValueError("`%s` argument is neither a sklearn `BaseEstimator` "
                         "instance nor a tensorflow Model instance. "
                         "Given argument, %s"%
                         (display_name, str(estimator)))
    return new_estimator


def check_network(network, copy=True,
                  name=None,
                  force_copy=False):
    """
    Check if the given network is a tensorflow Model.
    If ``copy`` is ``True``, a copy of the network is
    returned if possible.

    Parameters
    ----------
    network : tensorflow Model
        Network to check.

    copy : boolean (default=True)
        Whether to return a copy of the network or not.
        If cloning fail, a warning is raised.

    name : str (default="network")
        Name for the network.

    force_copy : boolean (default=False)
        If True, an error is raised if the cloning failed.
    """
    if not isinstance(network, Model):
        raise ValueError('Expected `network` argument '
                         'to be a `Model` instance, got: %s'%str(network))
    
    if copy:
        try:
            # TODO, be carefull of network with weights
            # but no input_shape
            if hasattr(network, "input_shape"):
                shape = network.input_shape[1:]
                new_network = clone_model(network, input_tensors=Input(shape))
                new_network.set_weights(network.get_weights())
            elif network.built:
                shape = network._build_input_shape[1:]
                new_network = clone_model(network, input_tensors=Input(shape))
                new_network.set_weights(network.get_weights())
            else:
                new_network = clone_model(network)
        except Exception as e:
            if force_copy:
                raise ValueError("`network` argument can't be duplicated. "
                                 "Recorded exception: %s. "%str(e))
            else:
                warnings.warn("`network` argument can't be duplicated. "
                              "Recorded exception: %s. "
                              "The current network will be used. "
                              "Use `copy=False` to hide this warning."%str(e))
                new_network = network
    else:        
        new_network = network
        
    if name is not None:
        new_network._name = name
    
    # Override the predict method to speed the prediction for small dataset
    new_network.predict = predict.__get__(new_network)
    return new_network


def get_default_encoder(name=None, state=None):
    """
    Return a tensorflow Model of one layer
    with 10 neurons and a relu activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential(name=name)
    model.add(Flatten())
    if state is not None:
        model.add(Dense(10, activation="relu",
        kernel_initializer=GlorotUniform(seed=state)))
    else:
        model.add(Dense(10, activation="relu"))
    return model


def get_default_task(activation=None, name=None, state=None):
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with linear
    activation.
    
    Parameters
    ----------
    activation : str (default=None)
        Final activation

    Returns
    -------
    tensorflow Model
    """
    model = Sequential(name=name)
    model.add(Flatten())
    if state is not None:
        model.add(Dense(10, activation="relu",
        kernel_initializer=GlorotUniform(seed=state)))
        model.add(Dense(10, activation="relu",
        kernel_initializer=GlorotUniform(seed=state)))
        model.add(Dense(1,
        kernel_initializer=GlorotUniform(seed=state)))
    else:
        model.add(Dense(10, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation=activation))
    return model


def get_default_discriminator(name=None, state=None):
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with sigmoid
    activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential(name=name)
    model.add(Flatten())
    if state is not None:
        model.add(Dense(10, activation="relu",
        kernel_initializer=GlorotUniform(seed=state)))
        model.add(Dense(10, activation="relu",
        kernel_initializer=GlorotUniform(seed=state)))
        model.add(Dense(1, activation="sigmoid",
        kernel_initializer=GlorotUniform(seed=state)))
    else:
        model.add(Dense(10, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
    return model


@tf.custom_gradient
def _grad_handler(x, lambda_):
    y = tf.identity(x)
    def custom_grad(dy):
        return (lambda_ * dy, 0. * lambda_)
    return y, custom_grad

class GradientHandler(Layer):
    """
    Multiply gradients with a scalar during backpropagation.

    Act as identity in forward step.
    
    Parameters
    ----------
    lambda_init : float (default=1.)
        Scalar multiplier
    """
    def __init__(self, lambda_init=1., name="g_handler"):
        super().__init__(name=name)
        self.lambda_init=lambda_init
        self.lambda_ = tf.Variable(lambda_init,
                                   trainable=False,
                                   dtype="float32")

    def call(self, x):
        """
        Call gradient handler.
        
        Parameters
        ----------
        x: object
            Inputs
            
        Returns
        -------
        x, custom gradient function
        """
        return _grad_handler(x, self.lambda_)


    def get_config(self):
        """
        Return config dictionnary.
        
        Returns
        -------
        dict
        """
        config = super().get_config().copy()
        config.update({
            'lambda_init': self.lambda_init
        })
        return config


def make_classification_da(n_samples=100, 
                           n_features=2,
                           random_state=2):
    """
    Generate a classification dataset for DA.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Size of source and target samples.
  
    n_features : int, optional (default=2)
        Number of features.
    
    random_state: int, optional (default=0)
        Random state number.
        
    Returns
    -------
    Xs : numpy array
        Source input data

    ys : numpy array
        Source output data
        
    Xt : numpy array
        Target input data

    yt : numpy array
        Target output data
    """
    np.random.seed(random_state)
    Xs, ys = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_features,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    Xt, yt = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_features,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    yt[:int(n_samples/2)] = 1; yt[int(n_samples/2):] = 0
    Xt[:, 0] += 1; Xt[:, 1] += 0.5;

    for i in range(n_features):
        Xs[:, i] = (Xs[:, i]-np.min(Xs[:, i]))/np.max(Xs[:, i]-np.min(Xs[:, i]))
        Xt[:, i] = (Xt[:, i]-np.min(Xt[:, i]))/np.max(Xt[:, i]-np.min(Xt[:, i]))
    
    return Xs, ys, Xt, yt


def make_regression_da(n_samples=100,
                       n_features=1,
                       random_state=0):
    """
    Generate a regression dataset for DA.
    
    Parameters
    ----------
    n_samples : int (default=100)
        Size of source and target samples.
        
    n_features : int (default=1)
        Sample dimension.
    
    random_state: int (default=0)
        Random state number.
        
    Returns
    -------
    Xs : numpy array
        Source input data

    ys : numpy array
        Source output data
        
    Xt : numpy array
        Target input data

    yt : numpy array
        Target output data
    """
    np.random.seed(random_state)
    
    Xs = np.random.uniform(size=(n_samples, n_features)) * 4 - 2
    Xs = np.sort(Xs)
    Xt = np.random.uniform(size=(n_samples, n_features)) * 2.5 + 2
    ys = (Xs[:, 0] + 0.1 * Xs[:, 0] ** 5 +
          np.random.randn(n_samples) * 0.2 + 1)
    yt = (Xt[:, 0] + 0.1 * (Xt[:, 0] - 2) **4  +
          np.random.randn(n_samples) * 0.4 + 1)
    
    Xt = (Xt - Xs.mean(0)) / Xs.std(0)
    yt = (yt - ys.ravel().mean()) / (2 * ys.ravel().std())
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0))
    ys = (ys - ys.ravel().mean()) / (2 * ys.ravel().std())
    
    X = np.concatenate((Xs, Xt))
    
    Xs = ((Xs - X.mean(0)) / X.std(0)) / 3
    Xt = ((Xt - X.mean(0)) / X.std(0)) / 3

    return Xs, ys, Xt, yt


def check_sample_weight(sample_weight, X):
    """
    Check sample weights.
    
    Parameters
    ----------
    sample_weight : array
        Sample weights.
        
    X : array
        Input array
        
    Returns
    -------
    sample_weight : array
    """
    if not sample_weight is None:
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
        )
        if len(sample_weight) != X.shape[0]:
            raise ValueError("`sample_weight` and X should have"
                             " same length, got %i, %i"%
                             (len(sample_weight), X.shape[0]))
        if np.any(sample_weight<0):
            raise ValueError("All weights from `sample_weight`"
                             " should be positive.")
        if sample_weight.sum() <= 0:
            sample_weight = np.ones(X.shape[0])
    return sample_weight


def set_random_seed(random_state):
    """
    Set random seed for numpy and 
    Tensorflow
    
    Parameters
    ----------
    random_state : int or None
        Random state, if None
        the current random generators
        remain unchanged
    """
    if random_state is None:
        pass
    else:
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        

def new_init(self, **kwargs):
    for k, v in self.__frozen_dict__.items():
        setattr(self, k, v)


def __deepcopy__(self, memo):
    return self     
        
        
def check_fitted_estimator(estimator):
    """
    Check Fitted Estimator
    
    This function is used to create a custom embedding
    on fitted estimator object in order to be able to
    clone them and keep its fitted arguments.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Fitted estimator
    
    Returns
    -------
    estimator : instance of "Fitted" + estimator class name
        Embedded fitted estimator
    """
    if "Fitted" == estimator.__class__.__name__[:6]:
        return estimator
    else:
        new_class = type("Fitted"+estimator.__class__.__name__,
                         (estimator.__class__,),
                         {"__init__": new_init,
                          "__frozen_dict__": {k: v for k, v in estimator.__dict__.items()}})
        return new_class()
    
    
def check_fitted_network(estimator):
    """
    Check Fitted Network
    
    Overwrite the ``__deepcopy__`` method from network
    such that deepcopy returns the same estimator.
    
    Parameters
    ----------
    estimator : tensorflow Model
        Fitted network
    
    Returns
    -------
    estimator : tensorflow Model
        Modified fitted network
    """
    if isinstance(estimator, Model):
        estimator.__deepcopy__ = __deepcopy__.__get__(estimator)
    return estimator
        
        
        
# Try to save the initial estimator if it is a Keras Model
# This is required for cloning the adapt method.
# if isinstance(self.estimator, Model):
#     self._has_keras_estimator = True
#     try:
#         self._deepcopy_estimator = check_estimator(estimator,
#                                                    copy=True,
#                                                    task=None,
#                                                    force_copy=True)
#     except BaseException as err:
#         if "The current network will be used" in str(err):
#             warnings.warn("The Tensorflow model used as estimator"
#                           " can't be deep copied. "
#                           "This may provoke some undesired behaviour"
#                           " when cloning the object.")
#         else:
#             raise
# else:
#     self._has_keras_estimator = False