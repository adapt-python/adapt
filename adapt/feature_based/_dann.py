"""
Discriminative Adversarial Neural Network
"""

import warnings

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from adapt.utils import (check_indexes,
                         check_network,
                         get_default_encoder,
                         get_default_task,
                         GradientReversal)


class _IncreaseLambda(Callback):
    """
    Callback for increasing trade-off parameter.
    """
    def __init__(self, lambdap, gamma):
        self.lambdap = lambdap
        self.gamma = gamma
        
    def on_epoch_end(self, epoch, logs={}):
        self.lambdap = 2 / (1 + K.exp(-self.gamma * epoch)) - 1


class DANN:
    """
    DANN: Discriminative Adversarial Neural Network
    
    DANN is a feature-based domain adaptation method.
    
    The goal of DANN is to find a new representation of the input features
    in which source and target data could not be distinguished by any
    **discriminator** network. This new representation is learned by an
    **encoder** network in an adversarial fashion. A **task** network is
    learned on the encoded space in parallel to the **encoder** and 
    **discriminator** networks.
    
    The three network paremeters are optimized according to the
    following objective function:
    
    .. math::
    
        \mathcal{L} = \mathcal{L}_{task}(F(\phi(X_S)), y_S) - 
        \lambda \\left(
        \mathcal{L}_{01}(D(\mathcal{R}(\phi(X_S))), \\textbf{0}) +
        \mathcal{L}_{01}(D(\mathcal{R}(\phi(X_T))), \\textbf{1})
        \\right)
        
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi, F, D` are respectively the **encoder**, the **task**
      and the **discriminator** networks
    - :math:`\\mathcal{R}` is the **reversal gradient layer** which inverses
      the gradient sign in back-propagation.
    - :math:`\lambda` is the trade-off parameter.
    
    The adversarial training is done through a **reversal gradient layer**
    placed between the **encoder** and the **discriminator** networks.
    This layer inverses the gradient sign in backpropagation, thus the
    two networks are optimized according to two opposite objective functions.
    
    The method has been originally introduced for **unsupervised**
    classification DA but it could be widen to other task in
    **supervised** DA straightforwardly.
    
    Parameters
    ----------
    get_encoder : callable, optional (default=None)
        Constructor for encoder network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a shallow network with 10 neurons is used
        as encoder network.
        
    get_task : callable, optional (default=None)
        Constructor for task network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network and an ``output_shape``
        argument giving the shape of the last layer.
        If ``None``, a shallow network is used as task network.
        
    get_discriminator : callable, optional (default=None)
        Constructor for discriminator network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a linear network is used as discriminator
        network.
        
    lambdap : float or None, optional (default=1.0)
        Trade-off parameter.
        If ``None``, ``lambdap`` increases gradually from 0 to 1
        according to the following formula:
        ``lambdap`` = 2/(1 + exp(-``gamma`` * p)) - 1.
        
    gamma : float, optional (default=10.0)
        Increase rate parameter.
        Characterized increase of trade-off parameter if
        ``lambdap`` is set to ``None``.
    
    enc_params : dict, optional (default=None)
        Additional arguments for ``get_encoder``.
        
    task_params : dict, optional (default=None)
        Additional arguments for ``get_task``.
        
    disc_params : dict, optional (default=None)
        Additional arguments for ``get_task``.
        
    compil_params : key, value arguments, optional
        Additional arguments for network compiler
        (loss, optimizer...).
        If none, loss is set to ``"binary_crossentropy"``
        and optimizer to ``"adam"``.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        Fitted encoder network.
        
    task_ : tensorflow Model
        Fitted task network.
        
    discriminator_ : tensorflow Model
        Fitted discriminator network.
    
    model_ : tensorflow Model
        Fitted model: the union of
        encoder, task and discriminator networks.
        
    References
    ----------
    .. [1] `[1] <http://jmlr.org/papers/volume17/15-239/15-239.pdf>`_ Y. Ganin, \
E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, \
and V. Lempitsky. "Domain-adversarial training of neural networks". In JMLR, 2016.
    """
    def __init__(self, get_encoder=None, get_task=None, get_discriminator=None,
                 lambdap=1.0, gamma=10.0, enc_params=None, task_params=None,
                 disc_params=None, **compil_params):
        self.get_encoder = get_encoder
        self.get_task = get_task
        self.get_discriminator = get_discriminator
        self.lambdap = lambdap
        self.gamma = gamma
        self.enc_params = enc_params
        self.task_params = task_params
        self.disc_params = disc_params
        self.compil_params = compil_params
        
        if self.get_encoder is None:
            self.get_encoder = get_default_encoder
        if self.get_task is None:
            self.get_task = get_default_task
        if self.get_discriminator is None:
            self.get_discriminator = get_default_task
        
        if self.enc_params is None:
            self.enc_params = {}
        if self.task_params is None:
            self.task_params = {}
        if self.disc_params is None:
            self.disc_params = {}

        
    def fit(self, X, y, src_index, tgt_index, tgt_index_labeled=None,
            **fit_params):
        """
        Fit DANN.

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
            Arguments given to the fit method of DANN model
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)
        
        if self.lambdap is None:
            self.lambdap = K.variable(0)
        
        self._create_model(X.shape[1:], y.shape[1:])
        
        if tgt_index_labeled is None:
            task_index = src_index
        else:
            task_index = np.concatenate((src_index, tgt_index_labeled))
        disc_index = np.concatenate((src_index, tgt_index))
        labels = np.array([0] * len(src_index) + [1] * len(tgt_index))
        
        max_size = max(len(disc_index), len(task_index))
        resized_task_ind = np.resize(task_index, max_size)
        resized_disc_ind = np.resize(disc_index, max_size)
        
        if self.lambdap is None:
            callback = _IncreaseLambda(self.lambdap, self.gamma)
            if "callbacks" in fit_prams:
                fit_prams["callbacks"].append(callback)
            else:
                fit_prams["callbacks"] = [callback]

        self.model_.fit([X[resized_task_ind], X[resized_disc_ind]],
                       [y[resized_task_ind], labels[resized_disc_ind]],
                       **fit_params)
        return self
    
    
    def _create_model(self, shape_X, shape_y):

        self.encoder_ = check_network(self.get_encoder, 
                            "get_encoder",
                            input_shape=shape_X,
                            **self.enc_params)
        self.task_ = check_network(self.get_task,
                            "get_task",
                            input_shape=self.encoder.output_shape[1:],
                            output_shape=shape_y,
                            **self.task_params)
        self.discriminator_ = check_network(self.get_discriminator,
                            "get_discriminator",
                            input_shape=self.encoder.output_shape[1:],
                            **self.disc_params)

        input_task = Input(shape_X)
        input_disc = Input(shape_X)

        encoded_task = self.encoder_(input_task)
        encoded_disc = self.encoder_(input_disc)

        tasked = self.task_(encoded_task)
        discrimined =  GradientReversal()(encoded_disc)
        discrimined = self.discriminator_(discrimined)

        self.model_ = Model([input_task, input_disc],
                           [tasked, discrimined], name="DANN")
        
        compil_params = self.compil_params.deepcopy()
        if "loss" in compil_params:
            task_loss = self.compil_params["loss"]
            compil_params.pop('loss')
        else:
            task_loss = "binary_crossentropy"
        
        if not "optimizer" in compil_params:
            compil_params["optimizer"] = "adam"
            
        if "loss_weights" in compil_params:
            compil_params.pop("loss_weights")
            warnings.warn("loss_weights compil param has been overwritten "
                          "by [1., lambdap]")
        
        lambdap = K.variable()
        
        self.model_.compile(loss=[task_loss, "binary_crossentropy"],
                            loss_weights=[1., self.lambdap],
                            **compil_params)

        return self


    def predict(self, X):
        """
        Return prediction of the task network on the encoded features.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        return self.task_.predict(self.encoder_.predict(X))
