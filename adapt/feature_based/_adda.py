"""
DANN
"""

import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Input, subtract
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from adapt.utils import (GradientHandler,
                         check_arrays,
                         check_network,
                         check_one_array)
from adapt.feature_based import BaseDeepFeature

EPS = K.epsilon()


class ADDA(BaseDeepFeature):
    """
    ADDA: Adversarial Discriminative Domain Adaptation

    ADDA is a feature-based domain adaptation method.
    
    The purpose of ADDA is to build a new feature representation
    in which source and target data could not be distinguished by
    any **discriminator** network. This feature representation is
    built with two **encoder** networks:
    
    - a **source encoder** trained to provide good features in order
      to learn the task on the source domain. The task is learned
      through a **task** network trained with the **source encoder**.
    - a **target encoder** trained to fool a **discriminator** network
      which tries to classify source and target data in the encoded space.
      The **target encoder** and the **discriminator** are trained
      in an adversarial fashion in the same way as GAN.
      
    The parameters of the four networks are optimized in a two stage
    algorithm where **source encoder** and **task** networks are first
    fitted according to the following optimization problem:
    
    .. math::
    
        \min_{\phi_S, F} \mathcal{L}_{task}(F(\phi_S(X_S)), y_S)
    
    In the second stage, **target encoder** and **discriminator**
    networks are fitted according to:
    
    .. math::
    
        \min_{\phi_T} & \; - \log(D(\phi_T(X_T)))) \\\\
        \min_{D} & \; - \log(D(\phi_S(X_S))) - \log(1 - D(\phi_T(X_T)))
    
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi_S, \phi_T, F, D` are respectively the **source encoder**,
      the **target encoder**, the **task** and the **discriminator** networks.
    
    The method has been originally introduced for **unsupervised**
    classification DA but it could be widen to other task in **supervised**
    DA straightforwardly.
    
    .. figure:: ../_static/images/adda.png
        :align: center
        
        Overview of the ADDA approach (source: [1])
    
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
        
    encoder_src : tensorflow Model (default=None)
        Source encoder netwok. A source encoder network can be
        given in the case of heterogenous features between
        source and target domains. If ``None``, a copy of the
        ``encoder`` network is used as source encoder.
        
    is_pretrained : boolean (default=False)
        Specify if the encoder is already pretrained on source or not

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
        
    optimizer_src : string or tensorflow optimizer (default=None)
        Optimizer of the source model. If ``None``, the source
        optimizer is a copy of ``optimizer``.
        
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
        and ``discriminator_`` networks.
        
    encoder_src_ : tensorflow Model
        Source encoder network
        
    model_src_ : tensorflow Model
        Fitted source model: the union of ``encoder_src_``
        and ``task_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    history_src_ : dict
        Source model history of the losses and metrics
        across the epochs. If ``yt`` is given in ``fit``
        method, target metrics and losses are recorded too.
        
    is_pretrained_ : boolean
        Specify if the encoder is already pretrained on
        source or not. If True, the ``fit`` method will
        only performs the second stage of ADDA.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import ADDA
    >>> np.random.seed(0)
    >>> Xs = np.concatenate((np.random.random((100, 1)),
    ...                      np.zeros((100, 1))), 1)
    >>> Xt = np.concatenate((np.random.random((100, 1)),
    ...                      np.ones((100, 1))), 1)
    >>> ys = 0.2 * Xs[:, 0]
    >>> yt = 0.2 * Xt[:, 0]
    >>> model = ADDA(random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_src_["task_t"][-1]
    0.0234...
    >>> model.history_["task_t"][-1]
    0.0009...
    
    See also
    --------
    DANN
    DeepCORAL
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1702.05464.pdf>`_ E. Tzeng, J. Hoffman, \
K. Saenko, and T. Darrell. "Adversarial discriminative domain adaptation". \
In CVPR, 2017.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 encoder_src=None,
                 is_pretrained=False,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 optimizer_src=None,
                 copy=True,
                 random_state=None):

        super().__init__(encoder, task, discriminator,
                         loss, metrics, optimizer, copy,
                         random_state)
        self.is_pretrained_ = is_pretrained
        
        if optimizer_src is None:
            self.optimizer_src = deepcopy(self.optimizer)
        else:
            self.optimizer_src = optimizer_src
        
        if encoder_src is None:
            self.encoder_src_ = check_network(self.encoder_,
                                              copy=True,
                                              display_name="encoder",
                                              force_copy=True,
                                              compile_=False)
            self.same_encoder_ = True
        else:
            self.encoder_src_ = check_network(encoder_src,
                                              copy=copy,
                                              display_name="encoder_src",
                                              compile_=False)
            self.same_encoder_ = False

        
    def fit_source(self, Xs, ys, Xt=None, yt=None, **fit_params):
        """
        Build and fit source encoder and task networks
        on source data.
        
        This method performs the first stage of ADDA.

        Parameters
        ----------
        Xs : numpy array
            Source input data.

        ys : numpy array
            Source output data.

        Xt : numpy array (default=None)
            Target input data. Target data are only
            used in the fit method of ``model_src_`` as
            validation data.
            
        yt : numpy array (default=None)
            Target output data. Target data are only
            used in the fit method of ``model_src_`` as
            validation data.

        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).

        Returns
        -------
        model_src_ : returns the fitted source model.
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
                
        # Call predict to avoid strange behaviour with
        # Sequential model whith unspecified input_shape
        zeros_enc_ = self.encoder_src_.predict(np.zeros((1,) + Xs.shape[1:]));
        self.task_.predict(zeros_enc_);
        
        self.model_src_ = Sequential()
        self.model_src_.add(self.encoder_src_)
        self.model_src_.add(self.task_)
        self.model_src_.compile(loss=self.loss_,
                                metrics=self.metrics_task_,
                                optimizer=self.optimizer_src)
        
        if (Xt is not None and yt is not None and 
            not "validation_data" in fit_params):
            hist = self.model_src_.fit(Xs, ys,
                                       validation_data=(Xt, yt),
                                       **fit_params)
            hist.history["task_t"] = hist.history.pop("val_loss")
            for k in hist.history:
                if "val_" in k:
                    hist.history[k.replace("val_", "") + "_t"] = hist.history.pop(k)
        else:
            hist = self.model_src_.fit(Xs, ys, **fit_params)

        hist.history["task_s"] = hist.history.pop("loss")
        
        for k, v in hist.history.items():
            if not hasattr(self, "history_src_"):
                self.history_src_ = {}
            self.history_src_[k] = self.history_src_.get(k, []) + v
        return self.model_src_


    def fit_target(self, Xs_enc, ys, Xt, yt=None, **fit_params):
        """
        Build and fit target encoder and discriminator
        networks on source data.
        
        This method performs the second stage of ADDA.

        Parameters
        ----------
        Xs_enc : numpy array
            Source encoded data.

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
        model_ : return the fitted target model.
        """
        self._fit(Xs_enc, ys, Xt, yt, **fit_params)
        return self.model_
        
    
    def fit(self, Xs, ys, Xt, yt=None, fit_params_src=None, **fit_params):
        """
        Perform the two stages of ADDA.
        
        First ``encoder_src_`` and ``task_`` are fitted using
        ``Xs`` and ``ys``. Then ``encoder_`` and ``discriminator_``
        are fitted using ``Xs``, ``Xt`` and ``ys``.
        
        Note that if fit is called again, only the training of
        ``encoder_`` and ``discriminator_`` is extended,
        ``encoder_src_`` and ``task_`` remaining as they are.
        
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

        fit_params_src : dict (default=None)
            Arguments given to the fit method of the
            source model (epochs, batch_size, callbacks...).
            If ``None``, fit_params_src is set to fit_params.

        fit_params : key, value arguments
            Arguments given to the fit method of the
            target model (epochs, batch_size, callbacks...).

        Returns
        -------
        self : returns an instance of self
        """
        if fit_params_src is None:
            fit_params_src = fit_params
        
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)
        
        if not self.is_pretrained_:
            self.fit_source(Xs, ys, Xt, yt, **fit_params_src)
            self.is_pretrained_ = True
            if self.same_encoder_:
                # Call predict to set architecture if no
                # input_shape is given
                self.encoder_.predict(np.zeros((1,) + Xt.shape[1:]))
                self.encoder_.set_weights(self.encoder_src_.get_weights())
        
        Xs_enc = self.encoder_src_.predict(Xs)
        
        self.fit_target(Xs_enc, ys, Xt, yt, **fit_params)
        return self
    
    
    def create_model(self, inputs_Xs, inputs_Xt):

        encoded_tgt = self.encoder_(inputs_Xt)
        encoded_tgt_nograd = GradientHandler(0.)(encoded_tgt)
        
        task_tgt = self.task_(encoded_tgt)

        disc_src = self.discriminator_(inputs_Xs)
        disc_tgt = self.discriminator_(encoded_tgt)
        disc_tgt_nograd = self.discriminator_(encoded_tgt_nograd)
        
        outputs = dict(disc_src=disc_src,
                       disc_tgt=disc_tgt,
                       disc_tgt_nograd=disc_tgt_nograd,
                       task_tgt=task_tgt)
        return outputs


    def get_loss(self, inputs_ys, inputs_yt, disc_src, disc_tgt,
                  disc_tgt_nograd, task_tgt):
        
        loss_disc = (-K.log(disc_src + EPS)
                     -K.log(1-disc_tgt_nograd + EPS))
        
        # The second term is here to cancel the gradient update on
        # the discriminator
        loss_enc = (-K.log(disc_tgt + EPS)
                    +K.log(disc_tgt_nograd + EPS))
        
        loss = K.mean(loss_disc) + K.mean(loss_enc)
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     disc_src, disc_tgt,
                     disc_tgt_nograd, task_tgt):
        metrics = {}
        
        disc = (-K.log(disc_src + EPS)
                -K.log(1-disc_tgt_nograd + EPS))
        
        metrics["disc"] = K.mean(disc)
        if inputs_yt is not None:
            task_t = self.loss_(inputs_yt, task_tgt)
            metrics["task_t"] = K.mean(task_t)
        
        names_task, names_disc = self._get_metric_names()
        
        if inputs_yt is not None:
            for metric, name in zip(self.metrics_task_, names_task):
                metrics[name + "_t"] = metric(inputs_yt, task_tgt)
                      
        for metric, name in zip(self.metrics_disc_, names_disc):
            pred = K.concatenate((disc_src, disc_tgt), axis=0)
            true = K.concatenate((K.ones_like(disc_src),
                                  K.zeros_like(disc_tgt)), axis=0)
            metrics[name] = metric(true, pred)
        return metrics
    
    
    def predict_features(self, X, domain="tgt"):
        """
        Return the encoded features of X.
        
        Parameters
        ----------
        X : array
            Input data

        domain: str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            outputs of ``encoder_`` are returned.
            If domain is ``"src"`` or ``"source"``,
            outputs of ``encoder_src_`` are returned.
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        X = check_one_array(X)
        if domain in ["tgt", "target"]:
            return self.encoder_.predict(X)
        elif domain in ["src", "source"]:
            return self.encoder_src_.predict(X)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
        
        
    def predict(self, X, domain="tgt"):
        """
        Return predictions of the task network on the encoded features.
        
        Parameters
        ----------
        X : array
            Input data
            
        domain : str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            outputs of ``encoder_`` are used.
            If domain is ``"src"`` or ``"source"``,
            outputs of ``encoder_src_`` are used.
            
        Returns
        -------
        y_pred : array
            predictions of task network
        """
        X = check_one_array(X)
        return self.task_.predict(self.predict_features(X, domain))
    
    
    def predict_disc(self, X, domain="tgt"):
        """
        Return predictions of the discriminator on the encoded features.
        
        Parameters
        ----------
        X : array
            Input data
            
        domain : str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            outputs of ``encoder_`` are used.
            If domain is ``"src"`` or ``"source"``,
            outputs of ``encoder_src_`` are used.
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """
        X = check_one_array(X)
        return self.discriminator_.predict(self.predict_features(X, domain))