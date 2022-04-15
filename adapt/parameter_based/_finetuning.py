import tensorflow as tf
from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import check_fitted_network


@make_insert_doc(["encoder", "task"], supervised=True)
class FineTuning(BaseAdaptDeep):
    """
    FineTuning : finetunes pretrained networks on target data.
    
    A pretrained source encoder should be given. A task network,
    pretrained or not, should be given too.
    
    Finetuning train both networks. The task network can be
    fitted first using the ``pretrain`` parameter. The layers
    to train in the encoder can be set via the parameter ``training``.
    
    .. figure:: ../_static/images/regulartransfer.png
        :align: center
        
        Transferring parameters of a CNN pretrained on Imagenet
        (source: [1])
    
    Parameters
    ----------        
    training : bool or list of bool, optional (default=True)
        Weither to train the ``encoder`` or not.
        If a list is given, values from ``training`` are assigned
        successively to the ``trainable`` attribute of the
        ``encoder`` layers going from the last layer to the first one.
        If the length of ``training`` is smaller than the length of
        the ``encoder`` layers list, the last value of ``training`` 
        will be asigned to the remaining layers.
        
    pretrain : bool (default=False)
        If True, the task network is first trained alone on the outputs
        of the encoder.

    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
    
    task_ : tensorflow Model
        Network.
        
    history_ : dict
        history of the losses and metrics across the epochs
        of the network training.
    
    Examples
    --------
    >>> from adapt.utils import make_regression_da
    >>> from adapt.parameter_based import FineTuning
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> src_model = FineTuning(loss="mse", pretrain=False, random_state=0)
    >>> src_model.fit(Xs, ys, epochs=100, verbose=0)
    >>> print(src_model.score(Xt, yt))
    1/1 [==============================] - 0s 82ms/step - loss: 0.2645
    0.26447802782058716
    >>> tgt_model = FineTuning(encoder=src_model.encoder_, loss="mse", pretrain=True,
    ...                        pretrain__epochs=30, random_state=0)
    >>> tgt_model.fit(Xt[:3], yt[:3], epochs=100, verbose=0)
    >>> tgt_model.score(Xt, yt)
    1/1 [==============================] - 0s 78ms/step - loss: 0.1114
    0.11144775152206421
    
    References
    ----------
    .. [1] `[1] <https://hal.inria.fr/hal-00911179v1/document>`_ \
Oquab M., Bottou L., Laptev I., Sivic J. "Learning and \
transferring mid-level image representations using convolutional \
neural networks". In CVPR, 2014.
    """
    
    def __init__(self,
                 encoder=None,
                 task=None,
                 Xt=None,
                 yt=None,
                 training=True,
                 pretrain=False,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        encoder = check_fitted_network(encoder)
        task = check_fitted_network(task)
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
        
    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit FineTuning.
        
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
    
    
    def pretrain_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)

        # loss
        with tf.GradientTape() as task_tape:                       
            # Forward pass
            Xs_enc = self.encoder_(Xs, training=False)
            ys_pred = self.task_(Xs_enc, training=True)

            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))

            # Compute the loss value
            loss = tf.reduce_mean(self.task_loss_(ys, ys_pred))
            task_loss = loss + sum(self.task_.losses)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables

        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))

        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        return logs
    
    
    def train_step(self, data):
        # Pretrain
        if self.pretrain_:
            return self.pretrain_step(data)
        
        else:
            # Unpack the data.
            Xs, Xt, ys, yt = self._unpack_data(data)

            # loss
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:                       
                # Forward pass
                Xs_enc = self.encoder_(Xs, training=False)
                ys_pred = self.task_(Xs_enc, training=True)

                # Reshape
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))

                # Compute the loss value
                loss = tf.reduce_mean(self.task_loss_(ys, ys_pred))
                task_loss = loss + sum(self.task_.losses)
                enc_loss = loss + sum(self.encoder_.losses)

            # Compute gradients
            trainable_vars_task = self.task_.trainable_variables
            trainable_vars_enc = self.encoder_.trainable_variables

            gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
            gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
            self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))

            # Update metrics
            self.compiled_metrics.update_state(ys, ys_pred)
            self.compiled_loss(ys, ys_pred)
            # Return a dict mapping metric names to current value
            logs = {m.name: m.result() for m in self.metrics}
            return logs
        
        
    def _initialize_weights(self, shape_X):
        super()._initialize_weights(shape_X)
                
        nb_layers = len(self.encoder_.layers)
        
        if isinstance(self.training, bool):
            self.encoder_.trainable = self.training
        elif isinstance(self.training, list):
            self.training += [self.training[-1]] * (nb_layers-len(self.training))
            for i in range(nb_layers):
                if hasattr(self.encoder_.layers[i], "trainable"):
                    self.encoder_.layers[i].trainable = self.training[nb_layers-i-1]
        else:
            raise ValueError("`training` parameter should be"
                             " of type bool or list, got %s"%str(type(self.training)))

    
    def predict_disc(self, X):
        """
        Not used.
        """     
        pass