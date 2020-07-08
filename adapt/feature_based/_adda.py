"""
Adversarial Discriminative Domain Adaptation
"""



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
    - a **target encoder** trained to fools a **discriminator** network
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
    get_encoder : callable, optional (default=None)
        Constructor for source and target encoder networks.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, shallow networks with 10 neurons are used
        as encoder networks.
        
    get_task : callable, optional (default=None)
        Constructor for task network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a linear network is used as task network.
        
    get_discriminator : callable, optional (default=None)
        Constructor for discriminator network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a linear network is used as discriminator
        network.
    
    enc_params : dict, optional (default={})
        Additional arguments for ``get_encoder``.
        
    task_params : dict, optional (default={})
        Additional arguments for ``get_task``.
        
    disc_params : dict, optional (default={})
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
    .. [1] `[1] <https://arxiv.org/pdf/1702.05464.pdf>`_ E. Tzeng, J. Hoffman, \
K. Saenko, and T. Darrell. "Adversarial discriminative domain adaptation". \
In CVPR, 2017.
    """
    def __init__(self, get_encoder=None, get_task=None, get_discriminator=None,
                 enc_params={}, task_params={}, disc_params={}, **compil_params):
        self.get_encoder = get_encoder
        self.get_task = get_task
        self.get_discriminator = get_discriminator
        self.enc_params = enc_params
        self.task_params = task_params
        self.disc_params = disc_params
        self.compil_params = compil_params


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
        pass


    def predict(self, domain="target"):
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
        pass

