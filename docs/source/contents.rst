.. _adapt:

ADAPT
=====

The principle of classical machine learning is to use data from the past to learn a predictive
model for the future. However it proceed on the assumption that
future data will look the same as previous seen ones. In a mathematical formulation we
would say that the domain of previous and future data are the same and
follow the same distribution.

In most of real-world problems this assumption is not verified and the data on which we
want to use a machine learning model are very different from those used to build the model.

To handle this kind of issue, many **domain adaptation** methods have been developped recently.
These methods are used to improve machine learning models applied on one
domain called **target** by transferring information from one or more related domains
called **source**.

.. _adapt.feature_based:

:ref:`adapt.feature_based <adapt.feature_based>`: Feature-Based Methods
-----------------------------------------------------------------------

Features-based approaches consist in finding a "good" features representation that minimize
both domains divergence between source and target and the loss of our predictive function fT.

.. figure:: _static/images/feature_based.PNG
    :width: 800px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center


.. automodule: adapt.feature_based
	:no-members:
	:no-inherited-members:

.. currentmodule:: adapt

.. rubric:: Methods

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_based.FE
   
   
.. _adapt.instance_based:

:ref:`adapt.instance_based <adapt.instance_based>`: Instance-Based Methods
--------------------------------------------------------------------------

The general principle of this method is to re-weight labeled data from the source domain
in order to correct the marginal distribution difference between the source and the target
domains. The re-weighted source data are then directly used with the few target labeled
data to train a predictive function fT().


.. figure:: _static/images/instance_based.PNG
    :width: 800px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center


.. automodule: adapt.instance_based
	:no-members:
    :no-inherited-members:
	
	
	
.. _adapt.parameter_based:

:ref:`adapt.parameter_based <adapt.parameter_based>`: Parameter-Based Methods
-----------------------------------------------------------------------------

Parameters-based approach consist to transfer knowledge through shared parameters of
source and target predictive models fS and fT or by creating multiple source models fS and
optimally combining them to form fT .

.. figure:: _static/images/parameter_based.PNG
    :width: 800px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center

.. automodule: adapt.parameter_based
	:no-members:
    :no-inherited-members:
	
	
	
.. _adapt.utils:

:ref:`adapt.utils <adapt.utils>`: Utility Functions
---------------------------------------------------

This module contains utility functions used in the previous modules.

.. automodule: adapt.utils
	:no-members:
    :no-inherited-members:
	
.. currentmodule:: adapt

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_indexes
   utils.check_estimator
