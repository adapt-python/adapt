.. _regression:

Regression Examples
===================

You will find here the application of DA methods from the ADAPT package on a simple one 
dimensional DA regression problem.

First we import packages needed in the following. We will use ``matplotlib Animation`` tools in order to
get a visual understanding of the mselected methods:

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

Experimental Setup
------------------

We now set the synthetic regression DA problem using the `toy_regression <generated/adapt.utils.toy_regression.html>`_ 
function from ``adapt.utils``.

.. code-block:: python

	from adapt.utils import toy_regression

	X, y, src_index, tgt_index, tgt_index_labeled = toy_regression()
	
We define here a ``show`` function which we will use in the following to visualize the algorithms performances
on the toy problem.

.. code-block:: python
	
	def show(y_pred=None, ax=None):
		if ax is None:
			ax = plt.gca()
		ax.scatter(X[src_index], y[src_index], s=50, label="source")
		ax.scatter(X[tgt_index], y[tgt_index], s=50, alpha=0.5, label="target")
		ax.scatter(X[tgt_index_labeled], y[tgt_index_labeled], s=100,
		c="black", marker="s", label="target labeled")
		if y_pred is not None:
			ax.plot(np.linspace(-0.7, 0.6, 100), y_pred, c="red", lw=3, label="predictions")
		ax.set_xlim((-0.7,0.6))
		ax.set_ylim((-1.3, 2.2))
		ax.legend()
		return ax

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=20, metadata=dict(artist='Adapt'), bitrate=1800)

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	show(ax=ax)
	plt.show()
	
.. figure:: _static/images/regression_setup.PNG
    :align: center
    :alt: alternate text
    :figclass: align-center

Notice that we also define a ``writer`` which will be used to record the evolution of predictions through epochs.

As we can see in the figure above (plotting the output data ``y`` with respect to the inputs ``X``),
source and target data define two distinct domains. We have modeled here a classical supervised 
DA issue where the goal is to build a good model on orange data knowing only the labels (``y``) of the blue
and black points.

We now define the base model used to learn the task. We use here a neural network with two hidden layer.
We also define a ``SavePrediction`` callback in order to save the prediction of the neural network at
each epoch.

.. code-block:: python
	
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Input, Dense, Reshape
	from tensorflow.keras.optimizers import Adam
	
	def get_base_model(input_shape=(1,), output_shape=(1,)):
		model = Sequential()
		model.add(Dense(100, activation='elu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(np.prod(output_shape)))
		model.compile(optimizer=Adam(0.01), loss='mean_squared_error')
		return model
	

.. code-block:: python
	
	from tensorflow.keras.callbacks import Callback
	
	class SavePrediction(Callback):  
	
		def __init__(self, X):
			self.X = X
			super().__init__()
				
		def on_epoch_end(self, batch, logs={}):
			if not hasattr(self.model, "custom_history_"):
				self.model.custom_history_ = []
			predictions = self.model.predict_on_batch(self.X).ravel()
			self.model.custom_history_.append(predictions)
				
	callbacks = [SavePrediction(np.linspace(-0.7, 0.6, 100).reshape(-1, 1))]


Traget Only
-----------

First, let's fit a network only on the three labeled target data. As we could have guessed,
this is not sufficient to build an efficient model on the whole target domain.

.. code-block:: python
	
	model = KerasRegressor(get_base_model, callbacks=callbacks, epochs=100, batch_size=64, verbose=0)
	model.fit(X[tgt_index_labeled], y[tgt_index_labeled]);


.. code-block:: python
	
	def animate(i):
		plt.clf()
		y_pred = model.model.custom_history_[i].ravel()
		show(y_pred)
	
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate(0)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
	ani.save('tgtOnly.mp4', writer=writer)

.. figure:: _static/images/tgtOnly.gif
    :align: center
    :alt: alternate text
    :figclass: align-center


Source Only
-----------

We would like to use the large amount of labeled source data to improve
the training of the neural network on the target domain. However,
as we can see on the figure below, using only the source
dataset fails to provide an efficient model.

.. code-block:: python
	
	model = KerasRegressor(get_base_model, callbacks=callbacks, epochs=100, batch_size=100, verbose=0)
	model.fit(X[src_index], y[src_index]);


.. code-block:: python
	
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate(0)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
	ani.save('srcOnly.mp4', writer=writer)

.. figure:: _static/images/srcOnlyReg.gif
    :align: center
    :alt: alternate text
    :figclass: align-center


All
---

Same thing happen when using both source and target labeled data. As the source sample ovewhelms the target one,
the model is not fitted enough on the target domain.

.. code-block:: python
	
	model = KerasRegressor(get_base_model, callbacks=callbacks, epochs=100, batch_size=110, verbose=0)
	model.fit(X[np.concatenate((src_index, tgt_index_labeled))],
	y[np.concatenate((src_index, tgt_index_labeled))]);

.. code-block:: python
	
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate(0)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
	ani.save('all.mp4', writer=writer)

.. figure:: _static/images/all.gif
    :align: center
    :alt: alternate text
    :figclass: align-center


CORAL
-----

Let's now consider the domain adaptation method `CORAL <generated/adapt.feature_based.CORAL.html>`_. 
This "two-stage" method first perfroms a feature alignment on source data and then fit
an estimator on the new feature space.

.. code-block:: python
	
	from adapt.feature_based import CORAL
	
	model = CORAL(get_base_model, lambdap=0)
	model.fit(X.reshape(-1, 1), y, src_index, tgt_index, tgt_index_labeled,
			  callbacks=callbacks, epochs=100, batch_size=110, verbose=0);
	X_transformed = model.get_features(X[src_index].reshape(-1, 1)).ravel()

.. code-block:: python
	
	def show_coral(y_pred=None, ax=None):
		if ax is None:
			ax = plt.gca()
		ax.scatter(X_transformed[src_index], y[src_index], s=50, label="source")
		ax.scatter(X[tgt_index], y[tgt_index], s=50, alpha=0.5, label="target")
		ax.scatter(X[tgt_index_labeled], y[tgt_index_labeled], s=100,
					c="black", marker="s", label="target labeled")
		if y_pred is not None:
			ax.plot(np.linspace(-0.7, 0.6, 100), y_pred, c="red", lw=3, label="predictions")
		ax.set_xlim((-0.7,0.6))
		ax.set_ylim((-1.3, 2.2))
		ax.legend()
		return ax
	
	def animate_coral(i):
		plt.clf()
		y_pred = model.estimator_.custom_history_[i].ravel()
		show_coral(y_pred)
		
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate_coral(99)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate_coral, frames=100, repeat=False)
	ani.save('coral.mp4', writer=writer)

.. figure:: _static/images/coral.gif
    :align: center
    :alt: alternate text
    :figclass: align-center


As we can see. when using CORAL method, source input data are translated closer to
target data. However, for this example, this is not enough to obtain a good model
on the target domain.


TrAdaBoostR2
------------

We now consider an instance-based method: `TrAdaBoostR2 <generated/adapt.instance_based.TrAdaBoostR2.html>`_.
This method consists in a reverse boosting algorithm decreasing the weights of source data poorly predicted
at each boosting iteraton.

.. code-block:: python
	
	from adapt.instance_based import TrAdaBoostR2
	
	model = TrAdaBoostR2(get_base_model, n_estimators=10)
	model.fit(X.reshape(-1, 1), y, src_index, tgt_index_labeled,
	callbacks=callbacks, epochs=100, batch_size=110, verbose=0);

.. code-block:: python
	
	def show_tradaboost(y_pred=None, weights_src=50, weights_tgt=50, ax=None):
		if ax is None:
			ax = plt.gca()
		ax.scatter(X[src_index], y[src_index], s=weights_src, label="source")
		ax.scatter(X[tgt_index], y[tgt_index], s=50, alpha=0.5, label="target")
		ax.scatter(X[tgt_index_labeled], y[tgt_index_labeled], s=weights_tgt,
					c="black", marker="s", alpha=0.5, label="target labeled")
		if y_pred is not None:
			ax.plot(np.linspace(-0.7, 0.6, 100), y_pred, c="red", lw=3, label="predictions")
		ax.set_xlim((-0.7,0.6))
		ax.set_ylim((-1.3, 2.2))
		ax.legend()
		return ax
	
	def animate_tradaboost(i):
		plt.clf()
		j = int(i / 100)
		i = i % 100
		y_pred = model.estimators_[j].custom_history_[i].ravel()
		weights_src = 10000 * model.sample_weights_src_[j]
		weights_tgt = 10000 * model.sample_weights_tgt_[j]
		show_tradaboost(y_pred, weights_src, weights_tgt)
		
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate_tradaboost(999)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate_tradaboost, frames=1000, repeat=False)
	ani.save('tradaboost.mp4', writer=writer)
	
	
.. figure:: _static/images/tradaboost.gif
    :align: center
    :alt: alternate text
    :figclass: align-center
	
	
As we can see on the figure above, `TrAdaBoostR2 <generated/adapt.instance_based.TrAdaBoostR2.html>`_ perfroms very well
on this toy DA issue! The importance weights are described by the size of data points.
We observe that the weights of source instances close to 0 are decreased as the weights of target instances increase.
This source instances indeed misleaded the fitting of the network on the target domain. Decreasing their weights helps
then a lot to obtain a good target model.



RegularTransfer
---------------

Finally, we consider here the paremeter-based method `RegularTransferNN <generated/adapt.parameter_based.RegularTransferNN.html>`_.
This method fits the target labeled data with a regularized loss. During training, the mean squared error on target data is
regularized with the euclidean distance between the target model parameters and the ones of a pre-trained source model.

.. code-block:: python
	
	from adapt.parameter_based import RegularTransferNN
	
	model = RegularTransferNN(get_base_model, lambdas=1.0)
	model.fit(X.reshape(-1, 1), y, src_index, tgt_index_labeled,
	callbacks=callbacks, epochs=100, batch_size=110, verbose=0);

.. code-block:: python
	
	def animate_regular(i):
		plt.clf()
		if i < 100:
			i = i % 100
			y_pred = model.model_src_.custom_history_[i].ravel()
		else:
			i = i % 100
			y_pred = model.model_tgt_.custom_history_[i].ravel()
		show(y_pred)
	
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	animate_regular(199)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate_regular, frames=200, repeat=False)
	ani.save('regular.mp4', writer=writer)
	
	
.. figure:: _static/images/regular.gif
    :align: center
    :alt: alternate text
    :figclass: align-center
	
See also
--------

:ref:`Classification Examples <classification>`


