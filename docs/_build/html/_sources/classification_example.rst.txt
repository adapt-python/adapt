.. _classification:

Classification Examples
=======================

You will find here the application of DA methods from the ADAPT package on a simple two 
dimensional DA classification problem.

First we import packages needed in the following. We will use ``matplotlib Animation`` tools in order to
get a visual understanding of the mselected methods:

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

Experimental Setup
------------------

We now set the synthetic classification DA problem using the `toy_classification <generated/adapt.utils.toy_classification.html>`_ 
function from ``adapt.utils``.

.. code-block:: python
	
	from adapt.utils import toy_classification
	
	X, y, src_index, tgt_index, tgt_index_labeled = toy_classification()

We define here ``show`` function which we will use in the following to visualize the algorithms performances
on the toy problem. We also define a ``get_predict_line`` function plotting the classes border.

.. code-block:: python
	
	def get_predict_line(model):
		line = np.linspace(0, 1, 100)
		grid = np.meshgrid(line, line)
		grid = np.concatenate(
				(grid[1].ravel().reshape(-1, 1),
				 grid[0].ravel().reshape(-1, 1)),
			   axis=1)
		y_pred = model.predict(grid).ravel()
		y_pred = (y_pred - 0.5 > 0).astype(int)
		
		line_y = []
		line_x = []
		for i in range(99):
			pred_i = y_pred[100 * i: 100 * (i + 1)]
			a = 0
			b = 99
			cut = None
			while pred_i[a] != pred_i[b] and np.abs(b-a) > 1:
				c = int((a+b) / 2)
				if pred_i[c] == pred_i[a]:
					cut = c
					a = c
				else:
					cut = c
					b = c
			if cut is not None:
				line_x.append(grid[100 * i + cut, 0])
				line_y.append(grid[100 * i + cut, 1])
		
		line_x = np.array(line_x)
		line_y = np.array(line_y)
		return line_x, line_y
	
	
	def show(X, lines=None, weights_src_min=50, weights_src_plus=50):
		ax = plt.gca()
		if lines is not None:
			ax.plot(lines[0], lines[1], c="red", lw=2, label="class boundary")
		ax.scatter(X[src_index][:50,0], X[src_index][:50, 1],
				   s=weights_src_min, marker='_', c='blue', label = 'Source')
		ax.scatter(X[src_index][50:,0], X[src_index][50:, 1],
				   s=weights_src_plus, marker='+', c='blue', label = 'Source')
		ax.scatter(X[tgt_index][50:,0], X[tgt_index][50:, 1], s=50, marker='_', c='orange', label = 'Target')
		ax.scatter(X[tgt_index][:50,0], X[tgt_index][:50, 1], s=50, marker='+', c='orange', label = 'Target')
		
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=20, metadata=dict(artist='Adapt'), bitrate=1800)

	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	show(X)
	plt.show()
	
.. figure:: _static/images/classification_setup.PNG
    :width: 600px
	:height: 450px
	:align: center
    :alt: alternate text
    :figclass: align-center

Notice that we also define a ``writer`` which will be used to record the evolution of predictions through epochs.

As we can see in the figure above (plotting the two dimensions of the input data),
source and target data define two distinct domains. We have modeled here a classical unsupervised 
DA issue where the goal is to build a good model on orange data knowing only the labels ("+" or "-" given by ``y``) of the blue
points.

We now define the base model used to learn the task. We use here a neural network with two hidden layer.
We also define a ``SavePrediction`` callback in order to save the prediction of the neural network at
each epoch.

.. code-block:: python
	
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Input, Dense, Reshape
	from tensorflow.keras.optimizers import Adam
	
	def get_base_model(input_shape=(2,), output_shape=(1,)):
		model = Sequential()
		model.add(Dense(100, activation='elu',
						input_shape=input_shape))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(optimizer=Adam(0.01), loss='binary_crossentropy')
		return model
	

.. code-block:: python
	
	from tensorflow.keras.callbacks import Callback
	
	class SavePrediction(Callback):  
		"""
		Callbacks which stores delimitation line at each epoch.
		"""
		def __init__(self):
			super().__init__()
				
		def on_epoch_end(self, batch, logs={}):
			"""Applied at the end of each epoch"""
			if not hasattr(model, "lines_x_"):
				model.lines_x_ = []
				model.lines_y_ = []
			line_x, line_y = get_predict_line(model)
			model.lines_x_.append(line_x)
			model.lines_y_.append(line_y)
			
	callbacks = [SavePrediction()]


Source Only
-----------

First, let's fit a network on source data without any adaptation. As we can observe,
the "-" labels from the target domain are missclassified.
Because of the "+" blue points close to the "-" domain, the network learns a classes
border not regularized enough and then misclassifies the target "-" data.

.. code-block:: python
	
	model = KerasClassifier(get_base_model,
    callbacks=callbacks, epochs=100, batch_size=110, verbose=0)
	model.fit(X[src_index], y[src_index]);


.. code-block:: python
	
	def animate(i):
		plt.clf()
		lines = (model.lines_x_[i], model.lines_y_[i])
		show(X, lines)
	
	%matplotlib notebook
	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	animate(99)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
	ani.save('srcOnly.mp4', writer=writer)

.. figure:: _static/images/srcOnlyCla.gif
    :align: center
    :alt: alternate text
    :figclass: align-center



mSDA
----

Let's now consider the domain adaptation method `mSDA <generated/adapt.feature_based.mSDA.html>`_. 
This "two-stage" method first perfroms a feature encoding on source data and then fits
an estimator using the new feature space.

The encoded features are learned with a stacked denoising autoencoder. Here we choose to reduce
the feature space to one feature with the encoder. 

.. code-block:: python
	
	def get_encoder(input_shape):
		model = Sequential()
		model.add(Dense(100, activation='elu',
						input_shape=input_shape))
		model.add(Dense(1, activation=None))
		model.compile(optimizer=Adam(0.01), loss='mse')
		return model
	
	def get_decoder(input_shape, output_shape):
		model = Sequential()
		model.add(Dense(100, activation='elu'))
		model.add(Dense(np.prod(output_shape), activation="sigmoid"))
		model.add(Reshape(output_shape))
		model.compile(optimizer=Adam(0.01), loss='mse')
		return model

.. code-block:: python
	
	from tensorflow.keras.callbacks import Callback
	
	class SaveEncoding(Callback):
		
		def __init__(self, X):
			self.X = X
			super().__init__()
				
		def on_epoch_end(self, batch, logs={}):
			if not hasattr(model, "encoding_"):
				model.encoding_ = []
			encoding = model.encoder_.predict(self.X)
			model.encoding_.append(encoding)
			
	callbacks_ae = [SaveEncoding(X)]
	
.. code-block:: python
	
	from adapt.feature_based import mSDA
	
	model = mSDA(get_encoder, get_decoder, get_base_model,
		noise_lvl=0.1, est_params=dict(input_shape=(1,)))
	
	model.fit(X, y, src_index, tgt_index,
		fit_params_ae = dict(
		callbacks=callbacks_ae, epochs=100, batch_size=200, verbose=0),
		callbacks=callbacks, epochs=100, batch_size=100, verbose=0);

.. code-block:: python	
	
	noise = np.random.randn(200, 1)*0.1
	def animate_pred(i):
		lines = (model.lines_x_[i], model.lines_y_[i])
		show(X, lines)
	
	def animate_encoding(i):
		X_enc = model.encoding_[i].reshape(-1, 1)
		X_enc = np.concatenate((noise, X_enc), 1)
		show(X_enc)
		
	def animate_msda(i):
		plt.sca(ax1)
		ax1.clear()
		ax1.set_title("Input Space")
		animate_pred(i)
		plt.sca(ax2)
		ax2.clear()
		ax2.set_title("Encoded Space")
		animate_encoding(i)
		
	fig, (ax1 , ax2) = plt.subplots(1, 2, figsize=(16, 6))
	animate_msda(0)
	
	ani = animation.FuncAnimation(fig, animate_msda, frames=100, repeat=False)
	ani.save('mSDA.mp4', writer=writer)

.. figure:: _static/images/msda.gif
    :width: 900px
	:height: 350px
	:align: center
    :alt: alternate text
    :figclass: align-center

We plot on the left, the evolution of the delimiting line through epochs. On the right
we represent the one dimensional encoded space (on the y axis), we give random x coordinate
to the inputs in order to get a better visualization.

As we can see, on the encoded feature space blue and orange "+" labels go on one side and "-" on 
the other. So when fitting the classifier on the encoded space using blue data, the network learns a
good delimitation line for both domains. Thus `mSDA <generated/adapt.feature_based.mSDA.html>`_
perfroms an efficient adaptation between domains for this toy DA issue.


DANN
----

We now consider the `DANN <generated/adapt.feature_based.DANN.html>`_ method.
This method consists in learning a new feature representation on which no 
``discriminator`` network can be able to classify between source and target data.

This is done with adversarial techniques following the principle of GANs.

.. code-block:: python
	
	def get_encoder(input_shape):
		model = Sequential()
		model.add(Dense(100, activation='elu',
						input_shape=input_shape))
		model.add(Dense(2, activation=None))
		model.compile(optimizer=Adam(0.01), loss='mse')
		return model
	
	def get_task(input_shape, output_shape):
		model = Sequential()
		model.add(Dense(10, activation='elu'))
		model.add(Dense(np.prod(output_shape), activation="sigmoid"))
		model.add(Reshape(output_shape))
		model.compile(optimizer=Adam(0.01), loss='mse')
		return model
	
	def get_discriminator(input_shape):
		model = Sequential()
		model.add(Dense(10, activation='elu'))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(optimizer=Adam(0.01), loss='mse')
		return model


.. code-block:: python
	
	from adapt.feature_based import DANN
	
	model = DANN(get_encoder, get_task, get_discriminator, lambdap=2.0, optimizer=Adam(0.01))
	model.fit(X, y, src_index, tgt_index,
		callbacks=[SavePrediction(), SaveEncoding(X)],
		epochs=100, batch_size=100, verbose=1);

.. code-block:: python
	
	noise = np.random.randn(200, 1)*0.1
	def animate_pred(i):
		lines = (model.lines_x_[i], model.lines_y_[i])
		show(X, lines)
	
	def animate_encoding(i):
		X_enc = model.encoding_[i]
		show(X_enc)
		
	def animate_dann(i):
		plt.sca(ax1)
		ax1.clear()
		ax1.set_title("Input Space")
		animate_pred(i)
		plt.sca(ax2)
		ax2.clear()
		ax2.set_title("Encoded Space")
		animate_encoding(i)
		
	fig, (ax1 , ax2) = plt.subplots(1, 2, figsize=(16, 6))
	animate_dann(99)
	
	ani = animation.FuncAnimation(fig, animate_dann, frames=100, repeat=False)
	ani.save('DANN.mp4', writer=writer)
	
	
.. figure:: _static/images/dann.gif
    :width: 900px
	:height: 350px
	:align: center
    :alt: alternate text
    :figclass: align-center
	
	
As we can see on the figure above, when applying `DANN <generated/adapt.feature_based.DANN.html>`_
algorithm, source data are projected on target data in the encoded space. Thus a ``task`` network
trained in parallel to the ``encoder`` and the ``discriminator`` is able to well classify "+" from "-" in the target domain.

KMM
---

Finally, we consider here the instance-based method `KMM <generated/adapt.instance_based.KMM.html>`_.
This method consists in reweighting source instances in order to minimize the MMD distance between
source and target domain. Then the algorithm trains a classifier using the reweighted source data.

.. code-block:: python
	
	from tensorflow.keras.callbacks import Callback
	
	class SavePredictionKMM(Callback):  
	
		def __init__(self):
			super().__init__()
				
		def on_epoch_end(self, batch, logs={}):
			if not hasattr(model, "lines_x_"):
				model.lines_x_ = []
				model.lines_y_ = []
			line_x, line_y = get_predict_line(model.estimator_)
			model.lines_x_.append(line_x)
			model.lines_y_.append(line_y)

.. code-block:: python

	from adapt.instance_based import KMM

	model = KMM(get_base_model, kernel_params=dict(gamma=0.1))
	model.fit(X, y, src_index, tgt_index,
		callbacks=[SavePredictionKMM()],
		epochs=100, batch_size=100, verbose=0);

.. code-block:: python
	
	def animate_kmm(i):
		plt.clf()
		lines = (model.lines_x_[i], model.lines_y_[i])
		weights_src = model.weights_ * 100
		show(X, lines, weights_src[:50], weights_src[50:])
	
	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	animate_kmm(99)
	plt.show()
	
	ani = animation.FuncAnimation(fig, animate_kmm, frames=100, repeat=False)
	ani.save('kmmClassif.mp4', writer=writer)
	
	
.. figure:: _static/images/kmm.gif
    :align: center
    :alt: alternate text
    :figclass: align-center
	
See also
--------

:ref:`Regression Examples <regression>`


