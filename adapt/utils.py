"""
Utility functions for adapt package.
"""

import warnings
import inspect

import numpy as np
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Reshape


def check_indexes(src_index, tgt_index, tgt_index_labeled=None):
    """
    Check indexes.

    Check that all given indexes are iterable. The function
    also raises warnings if similar indexes appear in both
    source and target index lists.

    Parameters
    ----------
    src_index : iterable
        Indexes of source instances.

    tgt_index : iterable
        Indexes of target instances.

    tgt_index_labeled : iterable, optional
        Indexes of labeled target instances.
    """
    list_index = [src_index, tgt_index]
    names = ["src_index", "tgt_index"]

    if tgt_index_labeled is not None:
        list_index.append(tgt_index_labeled)
        names.append("tgt_index_labeled")

    for index, name in zip(list_index, names):
        if not hasattr(index, "__iter__"):
            raise ValueError("%s is not an iterable."%name)

    if len(set(src_index) & set(tgt_index)) > 0:
        warnings.warn("Similar indexes appear in both"
                      " src_index and tgt_index")

    if tgt_index_labeled is not None:
        if len(set(src_index) & set(tgt_index_labeled)) > 0:
            warnings.warn("Similar indexes appear in both"
                          " src_index and tgt_index_labeled")


def check_estimator(get_estimator, **kwargs):
    """
    Build and check estimator.

    Check that ``get_estimator`` is a callable or is a class.
    Then, build an estimator and check that it
    implements ``fit`` and ``predict`` methods.

    Parameters
    ----------
    get_estimator : object
        Constructor for the estimator.

    kwargs : key, value arguments, optional
        Additional arguments for the constructor.
    """
    if (hasattr(get_estimator, "__call__")
        or inspect.isclass(get_estimator)):
        try:
            estimator = get_estimator(**kwargs)
        except Exception as err_info:
            raise ValueError("Failed to build estimator with"
                             " 'get_estimator'. "
                             "Please provide a builder function wich "
                             "returns a valid estimator object or "
                             "check the given additional arguments. \n \n"
                             "Exception message: %s"%err_info)
        
    else:
        raise ValueError("get_estimator is neither a callable nor a class")

    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise ValueError("Built estimator does "
                         "not implement fit and predict methods")

    return estimator


def check_network(get_model, constructor_name="get_model", **kwargs):
    """
    Build and check network.

    Check that ``get_model`` is a callable.
    Then, build an estimator and check that it is an 
    instance of tensorflow Model.
    
    If ``input_shape`` is in kwargs, the function checks that
    the built estimator takes an ``input_shape`` arguments.
    
    If ``output_shape`` is in kwargs, the function checks that
    the built estimator takes an ``output_shape`` arguments.

    Parameters
    ----------
    get_model : object
        Constructor for the estimator.

    constructor_name: str, optional (default="get_model")
        Name of contructor variable.

    kwargs : key, value arguments, optional
        Additional arguments for the constructor.
    """
    if hasattr(get_model, "__call__"):
        if ("input_shape" in kwargs and not
            "input_shape" in inspect.getfullargspec(get_model)[0]):
            raise ValueError("Constructor '%s' must take "
                             "an 'input_shape' argument"%constructor_name)
        if ("output_shape" in kwargs and not
            "output_shape" in inspect.getfullargspec(get_model)[0]):
            raise ValueError("Constructor '%s' must take "
                             "an 'output_shape' argument"%constructor_name)
        try:
            model = get_model(**kwargs)
        except Exception as err_info:
            raise ValueError("Failed to build model with constructor '%s'. "
                             "Please provide a builder function wich "
                             "returns a compiled tensorflow Model or "
                             "check the given additional arguments. \n \n"
                             "Exception message: %s"%(constructor_name,
                             err_info))
    else:
        raise ValueError("'%s' is not a callable"%constructor_name)

    if not isinstance(model, Model):
        raise ValueError("Built model from '%s' is not "
                         "a tensorflow Model instance"%constructor_name)
    return model


def get_default_encoder(input_shape):
    """
    Return a compiled tensorflow Model of a shallow network
    with 10 neurons and a linear activation.

    Parameters
    ----------
    input_shape: tuple
        Network input_shape

    Returns
    -------
    tensorflow Model
    """
    inputs = Input(shape=input_shape)
    flattened = Flatten()(inputs)
    outputs = Dense(10)(flattened)
    model = Model(inputs, outputs)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def get_default_task(input_shape, output_shape=(1,)):
    """
    Return a compiled tensorflow Model of a linear network
    with a linear activation.

    Parameters
    ----------
    input_shape: tuple
        Network input_shape

    output_shape: tuple, optional (default=(1,))
        Network output_shape

    Returns
    -------
    tensorflow Model
    """
    inputs = Input(shape=input_shape)
    flattened = Flatten()(inputs)
    outputs = Dense(np.prod(output_shape))(flattened)
    outputs = Reshape(output_shape)(outputs)
    model = Model(inputs, outputs)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


@tf.custom_gradient
def _grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradientReversal(Layer):
    """
    Inverse sign of gradient during backpropagation.

    Act as identity in forward step.
    """
    def __init__(self):
        super().__init__()

    def call(self, x):
        """
        Call gradient reversal.
        
        Parameters
        ----------
        x: object
            Inputs
            
        Returns
        -------
        func: gradient reversal function.
        """
        return _grad_reverse(x)


def toy_classification(n_samples=100, n_target_labeled=0,
                       n_features=2, random_state=2):
    """
    Generate toy classification dataset for DA.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Size of source and target samples.
        
    n_target_labeled : int, optional (default=3)
        Size of target labeled sample.
  
    n_features : int, optional (default=2)
        Number of features.
    
    random_state: int, optional (default=0)
        Random state number.
        
    Returns
    -------
    X : numpy array
        Input data

    y : numpy array
        Labels
        
    src_index : numpy array
        Source indexes.
    
    tgt_index : numpy array
        Target indexes.
    
    tgt_index_labeled : numpy array
        Target indexes labeled. 
    """
    np.random.seed(random_state)
    Xs, ys = make_classification(n_samples=100, n_features=2, n_informative=2,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    Xt, yt = make_classification(n_samples=100, n_features=2, n_informative=2,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    yt[:50] = 1; yt[50:] = 0
    Xt[:, 0] += 1; Xt[:, 1] += 0.5;

    Xs[:, 0] = (Xs[:, 0]-np.min(Xs[:, 0]))/np.max(Xs[:, 0]-np.min(Xs[:, 0]))
    Xs[:, 1] = (Xs[:, 1]-np.min(Xs[:, 1]))/np.max(Xs[:, 1]-np.min(Xs[:, 1]))

    Xt[:, 0] = (Xt[:, 0]-np.min(Xt[:, 0]))/np.max(Xt[:, 0]-np.min(Xt[:, 0]))
    Xt[:, 1] = (Xt[:, 1]-np.min(Xt[:, 1]))/np.max(Xt[:, 1]-np.min(Xt[:, 1]))

    X = np.concatenate((Xs, Xt))
    y = np.concatenate((ys, yt))
    src_index = range(n_samples)
    tgt_index = range(n_samples, 2*n_samples)
    tgt_index_labeled = np.random.choice(n_samples,
                      n_target_labeled) + n_samples
    
    return X, y, np.array(src_index), np.array(tgt_index), tgt_index_labeled


def toy_regression(n_samples=100, n_target_labeled=3, random_state=0):
    """
    Generate toy regression dataset for DA.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Size of source and target samples.
        
    n_target_labeled : int, optional (default=3)
        Size of target labeled sample.
    
    random_state: int, optional (default=0)
        Random state number.
        
    Returns
    -------
    X : numpy array
        Input data

    y : numpy array
        Labels
        
    src_index : numpy array
        Source indexes.
    
    tgt_index : numpy array
        Target indexes.
    
    tgt_index_labeled : numpy array
        Target indexes labeled. 
    """
    np.random.seed(random_state)
    
    Xs = np.random.uniform(size=n_samples) * 4 - 2
    Xs = np.sort(Xs)
    Xt = np.random.uniform(size=n_samples) * 2.5 + 2
    ys = (Xs + 0.1 * Xs ** 5 +
          np.random.randn(n_samples) * 0.2 + 1)
    yt = (Xt + 0.1 * (Xt - 2) **4  +
          np.random.randn(n_samples) * 0.4 + 1)
    
    Xt = (Xt - Xs.ravel().mean()) / Xs.ravel().std()
    yt = (yt - ys.ravel().mean()) / (2 * ys.ravel().std())
    Xs = (Xs - Xs.ravel().mean()) / (Xs.ravel().std())
    ys = (ys - ys.ravel().mean()) / (2 * ys.ravel().std())
    
    X = np.concatenate((Xs, Xt))
    y = np.concatenate((ys, yt))
    
    X = ((X - X.ravel().mean()) / X.ravel().std()) / 3
    
    src_index = np.array(range(n_samples))
    tgt_index = np.array(range(n_samples, 2 * n_samples))
    tgt_index_labeled = np.random.choice(n_samples,
                            n_target_labeled) + n_samples

    return X, y, src_index, tgt_index, tgt_index_labeled