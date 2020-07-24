"""
Adversarial Discriminative Domain Adaptation
"""

import copy

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from adapt.utils import (check_indexes,
                         check_network,
                         get_default_encoder,
                         get_default_task,
                         GradientReversal)

class ADDA:
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
    
        \max_{\phi_T} \min_{D} \mathcal{L}_{01}(D(\phi_S(X_S)), \\textbf{0})
        + \mathcal{L}_{01}(D(\phi_T(X_T)), \\textbf{1})
    
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi_S, \phi_T, F, D` are respectively the **source encoder**,
      the **target encoder**, the **task** and the **discriminator** networks.
    
    The method has been originally introduced for **unsupervised**
    classification DA but it could be widen to other task in **supervised**
    DA straightforwardly.
    
    Parameters
    ----------
    get_src_encoder : callable, optional (default=None)
        Constructor for source encoder networks.
        The constructor should return a tensorflow compiled Model.
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, shallow networks with 10 neurons are used
        as encoder networks.
        
    get_tgt_encoder : callable, optional (default=None)
        Constructor for target encoder networks.
        The constructor should return a tensorflow compiled Model.
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, shallow networks with 10 neurons are used
        as encoder networks.
        
    get_task : callable, optional (default=None)
        Constructor for task network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network and an ``output_shape``
        argument giving the shape of the last layer.
        If ``None``, a linear network is used as task network.
        
    get_discriminator : callable, optional (default=None)
        Constructor for discriminator network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a linear network is used as discriminator
        network.
    
    src_enc_params : dict, optional (default=None)
        Additional arguments for ``get_src_encoder``.
        
    tgt_enc_params : dict, optional (default=None)
        Additional arguments for ``get_tgt_encoder``.
        
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
    src_encoder_ : tensorflow Model
        Fitted source encoder network.
        
    tgt_encoder_ : tensorflow Model
        Fitted source encoder network.
        
    task_ : tensorflow Model
        Fitted task network.
        
    discriminator_ : tensorflow Model
        Fitted discriminator network.
    
    src_model_ : tensorflow Model
        Fitted source model: the union of
        source encoder and task networks.
        
    tgt_model_ : tensorflow Model
        Fitted target model: the union of
        target encoder, task and discriminator networks.

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1702.05464.pdf>`_ E. Tzeng, J. Hoffman, \
K. Saenko, and T. Darrell. "Adversarial discriminative domain adaptation". \
In CVPR, 2017.
    """
    def __init__(self, get_src_encoder=None, get_tgt_encoder=None,
                 get_task=None, get_discriminator=None,
                 src_enc_params={}, tgt_enc_params={}, task_params={},
                 disc_params={}, **compil_params):
        self.get_src_encoder = get_src_encoder
        self.get_tgt_encoder = get_tgt_encoder
        self.get_task = get_task
        self.get_discriminator = get_discriminator
        self.src_enc_params = src_enc_params
        self.tgt_enc_params = tgt_enc_params
        self.task_params = task_params
        self.disc_params = disc_params
        self.compil_params = compil_params
        
        if self.get_src_encoder is None:
            self.get_src_encoder = get_default_encoder
        if self.get_tgt_encoder is None:
            self.get_tgt_encoder = get_default_encoder
        if self.get_task is None:
            self.get_task = get_default_task
        if self.get_discriminator is None:
            self.get_discriminator = get_default_task
            
        if self.src_enc_params is None:
            self.src_enc_params = {}
        if self.tgt_enc_params is None:
            self.tgt_enc_params = {}
        if self.task_params is None:
            self.task_params = {}
        if self.disc_params is None:
            self.disc_params = {}


    def fit(self, X, y, src_index, tgt_index, tgt_index_labeled=None,
            fit_params_src=None, **fit_params_tgt):
        """
        Fit ADDA.

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

        fit_params_src : dict, optional (default=None)
            Arguments given to the fit process of source encoder
            and task networks (epochs, batch_size...).
            If None, ``fit_params_src = fit_params_tgt``
        
        fit_params_tgt : key, value arguments
            Arguments given to the fit method of the ADDA model,
            i.e. fitting of target encoder and discriminator.
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)
        
        if fit_params_src is None:
            fit_params_src = fit_params_tgt
        
        if tgt_index_labeled is None:
            src_index_bis = src_index
        else:
            src_index_bis = np.concatenate((src_index, tgt_index_labeled))
                 
        self._create_model(X.shape[1:], y.shape[1:])
        
        max_size = max(len(src_index_bis), len(tgt_index))
        resize_tgt_ind = np.resize(tgt_index, max_size)
        resize_src_ind = np.resize(src_index_bis, max_size)
        
        self.src_model_.fit(X[src_index_bis], y[src_index_bis],
                            **fit_params_src)
        
        self.tgt_model_.fit([self.src_encoder_.predict(X[resize_src_ind]),
                            X[resize_tgt_ind]],
                            **fit_params_tgt)
        return self
    
    
    def _create_model(self, shape_X, shape_y):
        
        compil_params = copy.deepcopy(self.compil_params)
        if not "loss" in compil_params:
            compil_params["loss"] = "binary_crossentropy"        
        if not "optimizer" in compil_params:
            compil_params["optimizer"] = "adam"
        
        self.src_encoder_ = check_network(self.get_src_encoder,
                            "get_src_encoder",
                            input_shape=shape_X,
                            **self.src_enc_params)
        self.tgt_encoder_ = check_network(self.get_tgt_encoder,
                            "get_tgt_encoder",
                            input_shape=shape_X,
                            **self.tgt_enc_params)

        if self.src_encoder_.output_shape != self.tgt_encoder_.output_shape:
            raise ValueError("Target encoder output shape does not match "
                             "the one of source encoder.")

        self.task_ = check_network(self.get_task,
                            "get_task",
                            input_shape=self.src_encoder_.output_shape[1:],
                            output_shape=shape_y,
                            **self.task_params)
        self.discriminator_ = check_network(self.get_discriminator,
                            "get_discriminator",
                            input_shape=self.src_encoder_.output_shape[1:],
                            **self.disc_params)
        
        input_task = Input(shape_X)
        encoded_source = self.src_encoder_(input_task)
        tasked = self.task_(encoded_source)
        self.src_model_ = Model(input_task, tasked, name="ModelSource")
        self.src_model_.compile(**compil_params)
               
        input_source = Input(self.src_encoder_.output_shape[1:])
        input_target = Input(shape_X)
        encoded_target = self.tgt_encoder_(input_target)
        discrimined_target = GradientReversal()(encoded_target)
        discrimined_target = self.discriminator_(discrimined_target)
        discrimined_source = self.discriminator_(input_source)
        
        loss = (-K.mean(K.log(discrimined_target)) -
                K.mean(K.log(1 - discrimined_source)))
        
        self.tgt_model_ = Model([input_source, input_target],
                                [discrimined_source, discrimined_target],
                                name="ModelTarget")
        self.tgt_model_.add_loss(loss)
        
        compil_params.pop("loss")
        self.tgt_model_.compile(**compil_params)        
        return self


    def predict(self, X, domain="target"):
        """
        Return the predictions of task network on the encoded feature space.

        ``domain`` arguments specify how features from ``X``
        will be considered: as ``"source"`` or ``"target"`` features.
        If ``"source"``, source encoder will be used. 
        If ``"target"``, target encoder will be used.

        Parameters
        ----------
        X : array
            Input data.

        domain : str, optional (default="target")
            Choose between ``"source"`` and ``"target"`` encoder.

        Returns
        -------
        y_pred : array
            Prediction of task network.

        Notes
        -----
        As ADDA is an anti-symetric feature-based method, one should
        indicates the domain of ``X`` in order to apply the appropriate
        feature transformation.
        """
        if domain == "target":
            X = self.tgt_encoder_.predict(X)
        elif domain == "source":
            X = self.src_encoder_.predict(X)
        else:
            raise ValueError("Choose between source or target for domain name")
        return self.task_.predict(X)
