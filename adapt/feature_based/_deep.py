"""
Deep Feature-Based Models
"""

import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Input, subtract
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
import tensorflow.keras.backend as K

from adapt.utils import (GradientHandler,
                         check_arrays,
                         check_one_array,
                         check_network,
                         get_default_encoder,
                         get_default_task,
                         get_default_discriminator)

EPS = K.epsilon()

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


class UpdateLambda(Callback):
    """
    Callback updating the lambda trade-off
    of a LambdaFactor layer according to the
    following formula:
    
    lambda = 2 / (1 + exp(-gamma * p)) - 1
    
    With p incresing from 0 to 1 during the
    training process
    
    Parameters
    ----------
    gamma : float (default=10.)
        gamma factor

    lambda_name : string (default="lambda_layer")
        Name of the LambdaLayer instance to
        update.
    """
    def __init__(self, gamma=10., lambda_name="g_handler"):
        self.gamma = gamma
        self.lambda_name = lambda_name
        self.steps = 0.
       
    def on_train_batch_begin(self, batch, logs=None):
        progress = self.steps / self.total_steps
        self.model.get_layer(self.lambda_name).lambda_.assign(
            -K.cast(2. / (1. + K.exp(-self.gamma * progress)) - 1., "float32")
        )
        self.steps += 1.
        
    def set_params(self, params):
        self.total_steps = (params["epochs"] * params["steps"])


class BaseDeepFeature:
    """
    Base for Deep features-based methods.
    
    This object is used as basis for deep DA methods.
    The object takes three networks : ``encoder``, ``task`` and
    ``discriminator``. A deep DA object can be created as
    a ``BaseDeepFeature`` child by implementing the
    ``create_model`` and ``get_loss`` methods.
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    discriminator : tensorflow Model (default=None)
        Discriminator netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as discriminator
        network. Note that the output shape of the discriminator should
        be ``(None, 1)`` and a ``sigmoid`` activation should be used.

    loss : string or tensorflow loss (default="mse")
        Loss function used for the task.
        
    metrics : dict or list of string or tensorflow metrics (default=None)
        Metrics given to the model. If a list is provided,
        metrics are used on both ``task`` and ``discriminator``
        outputs. To give seperated metrics, please provide a
        dict of metrics list with ``"task"`` and ``"disc"`` as keys.
        
    optimizer : string or tensorflow optimizer (default=None)
        Optimizer of the model. If ``None``, the
        optimizer is set to tf.keras.optimizers.Adam(0.001)
        
    copy : boolean (default=True)
        Whether to make a copy of ``encoder``, ``task`` and
        ``discriminator`` or not.
        
    random_state : int (default=None)
        Seed of random generator.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
        
    discriminator_ : tensorflow Model
        discriminator network.
    
    model_ : tensorflow Model
        Fitted model: the union of ``encoder_``,
        ``task_`` and ``discriminator_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    See also
    --------
    DANN
    ADDA
    DeepCORAL
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
                
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        if encoder is None:
            encoder = get_default_encoder()
        if task is None:
            task = get_default_task()
        if discriminator is None:
            discriminator = get_default_discriminator()
        
        if not isinstance(metrics, (list, dict, type(None))):
            raise ValueError("`metrics` argument should be an instance "
                             "of dict or list")
        
        if isinstance(metrics, dict):
            metrics_disc = metrics.get("disc")
            metrics_task = metrics.get("task")
        else:
            metrics_disc = metrics
            metrics_task = metrics
        if metrics_disc is None:
            metrics_disc = []
        if metrics_task is None:
            metrics_task = []
        
        if optimizer is None:
            optimizer = Adam(0.001)
        
        self.encoder_ = check_network(encoder,
                                      copy=copy,
                                      display_name="encoder",
                                      compile_=False)
        self.task_ = check_network(task,
                                   copy=copy,
                                   display_name="task",
                                   compile_=False)
        self.discriminator_ = check_network(discriminator,
                                            copy=copy,
                                            display_name="discriminator",
                                            compile_=False)
        self.loss = loss
        self.metrics = metrics
        self.loss_ = tf.keras.losses.get(loss)
        self.metrics_disc_ = []
        self.metrics_task_ = []
        for m in metrics_disc:
            if ((isinstance(m, str) and "acc" in m) or 
                (hasattr(m, "__name__") and "acc" in m.__name__) or
                (hasattr(m, "__class__") and "Acc" in m.__class__.__name__)):
                self.metrics_disc_.append(accuracy)
            else:
                self.metrics_disc_.append(tf.keras.metrics.get(m))
        for m in metrics_task:
            if ((isinstance(m, str) and "acc" in m) or 
                (hasattr(m, "__name__") and "acc" in m.__name__) or
                (hasattr(m, "__class__") and "Acc" in m.__class__.__name__)):
                self.metrics_task_.append(accuracy)
            else:
                self.metrics_task_.append(tf.keras.metrics.get(m))
        self.optimizer = optimizer
        self.copy = copy
        self.random_state = random_state
        

    def _initialize_networks(self, shape_Xt):
        zeros_enc_ = self.encoder_.predict(np.zeros((1,) + shape_Xt));
        self.task_.predict(zeros_enc_);
        self.discriminator_.predict(zeros_enc_);
    
    
    def _build(self, shape_Xs, shape_ys,
                    shape_Xt, shape_yt):
        # Call predict to avoid strange behaviour with
        # Sequential model whith unspecified input_shape
        self._initialize_networks(shape_Xt)
                
        inputs_Xs = Input(shape_Xs)
        inputs_ys = Input(shape_ys)
        inputs_Xt = Input(shape_Xt)
                
        if shape_yt is None:
            inputs_yt = None
            inputs = [inputs_Xs, inputs_ys, inputs_Xt]
        else:
            inputs_yt = Input(shape_yt)
            inputs = [inputs_Xs, inputs_ys,
                      inputs_Xt, inputs_yt]
        
        outputs = self.create_model(inputs_Xs=inputs_Xs,
                                    inputs_Xt=inputs_Xt)
        
        self.model_ = Model(inputs, outputs)
        
        loss = self.get_loss(inputs_ys=inputs_ys,
                             inputs_yt=inputs_yt,
                              **outputs)
        metrics = self.get_metrics(inputs_ys=inputs_ys,
                                    inputs_yt=inputs_yt,
                                    **outputs)
        
        self.model_.add_loss(loss)
        for k in metrics:            
            self.model_.add_metric(tf.reduce_mean(metrics[k]),
                                   name=k, aggregation="mean")
        
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model_.compile(optimizer=self.optimizer)
        self.history_ = {}
        return self
    
    
    def _fit(self, Xs, ys, Xt, yt=None, **fit_params):
        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)
               
        shape_Xs = Xs.shape[1:]
        shape_Xt = Xt.shape[1:]
        shape_ys = ys.shape[1:]
        if yt is None:
            shape_yt = None
        else:
            shape_yt = yt.shape[1:]
            
        max_size = max(len(Xs), len(Xt))
        
        Xs = np.resize(Xs, (max_size,) + shape_Xs)
        ys = np.resize(ys, (max_size,) + shape_ys)
        Xt = np.resize(Xt, (max_size,) + shape_Xt)
        if yt is not None:
            yt = np.resize(yt, (max_size,) + shape_yt)
        
        if not hasattr(self, "model_"):
            self._build(shape_Xs, shape_ys,
                       shape_Xt, shape_yt)
        
        if yt is None:
            hist = self.model_.fit([Xs, ys, Xt],
                                   **fit_params)
        else:
            hist = self.model_.fit([Xs, ys, Xt, yt],
                                   **fit_params)

        for k, v in hist.history.items():
            self.history_[k] = self.history_.get(k, []) + v
        return self
    
    
    def _get_metric_names(self):
        names_task = []
        for metric, i in zip(self.metrics_task_,
                             range(len(self.metrics_task_))):
            if hasattr(metric, "__name__"):
                name = metric.__name__
            elif hasattr(metric, "__class__"):
                name = metric.__class__.__name__
            else:
                name = str(i)
            if "_" in name: 
                short_name = ""
                for s in name.split("_"):
                    if len(s) > 0:
                        short_name += s[0]
            else:
                short_name = name[:3]
            
            names_task.append("task_" + short_name)
            
        names_disc = []
        for metric, i in zip(self.metrics_disc_,
                             range(len(self.metrics_disc_))):
            if hasattr(metric, "__name__"):
                name = metric.__name__
            elif hasattr(metric, "__class__"):
                name = metric.__class__.__name__
            else:
                name = str(i)
            if "_" in name: 
                short_name = ""
                for s in name.split("_"):
                    if len(s) > 0:
                        short_name += s[0]
            else:
                short_name = name[:3]
            names_disc.append("disc_" + short_name)
        return names_task, names_disc


    def create_model(self, inputs_Xs, inputs_Xt):
        """
        Create model. 
        
        Give the model architecture from the Xs, Xt
        inputs to the outputs.
        
        Parameters
        ----------
        inputs_Xs : InputLayer
            Input layer for Xs entries.
            
        inputs_Xt : InputLayer
            Input layer for Xt entries.
        
        Returns
        -------
        outputs : dict of tf Tensors
            Outputs tensors of the model
            (used to compute the loss).
        """
        pass

    
    def get_loss(self, inputs_ys, inputs_yt, **ouputs):
        """
        Get loss.
        
        Parameters
        ----------
        inputs_ys : InputLayer
            Input layer for ys entries.
            
        inputs_yt : InputLayer
            Input layer for yt entries.
        
        outputs : dict of tf Tensors
            Model outputs tensors.
        
        Returns
        -------
        loss : tf Tensor
            Model loss
        """
        pass
   

    def get_metrics(self, inputs_ys, inputs_yt, **outputs):
        """
        Get Metrics.
        
        Parameters
        ----------
        inputs_ys : InputLayer
            Input layer for ys entries.
            
        inputs_yt : InputLayer
            Input layer for yt entries.
        
        outputs : dict of tf Tensors
            Model outputs tensors.
        
        Returns
        -------
        metrics : dict of tf Tensors
            Model metrics. dict keys give the
            name of the metric and dict values
            give the corresponding Tensor.
        """
        return {}


    def fit(self, Xs, ys, Xt, yt=None, **fit_params):
        """
        Fit Model. Note that ``fit`` does not reset
        the model but extend the training.

        Parameters
        ----------
        Xs : numpy array
            Source input data.

        ys : numpy array
            Source output data.

        Xt : numpy array
            Target input data.
            
        yt : numpy array (default=None)
            Target output data. `yt` is only used
            for validation metrics.

        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).

        Returns
        -------
        self : returns an instance of self
        """
        return self._fit(Xs, ys, Xt, yt, **fit_params)


    def predict(self, X):
        """
        Return predictions of the task network on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_pred : array
            predictions of task network
        """
        X = check_one_array(X)
        return self.task_.predict(self.predict_features(X))
    
    
    def predict_features(self, X):
        """
        Return the encoded features of X.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        X = check_one_array(X)
        return self.encoder_.predict(X)
    
    
    def predict_disc(self, X):
        """
        Return predictions of the discriminator on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """
        X = check_one_array(X)
        return self.discriminator_.predict(self.predict_features(X))