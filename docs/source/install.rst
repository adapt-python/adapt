.. _install:

Installation
============

Pypi Installation
-----------------

This package is available on `Pypi <https://badge.fury.io/py/adaptation>`_. It has been tested on Linux, MacOSX and Windows
for Python versions: 3.5, 3.6 and 3.7. It can be installed with the following command line:

.. code-block:: python
	
	pip install adaptation

The following dependencies are required and will be installed with the library:

- numpy
- scipy
- tensorflow (>= 2.0)
- scikit-learn

If for some reason, these packages failed to install, you can do it manually with:

.. code-block:: python

	pip install numpy scipy tensorflow scikit-learn

Finally import the module in your python scripts with:

.. code-block:: python

	import adapt
	
Examples
--------

Please look at these examples to learn how to use the algorithms provided by the ADAPT package.

:ref:`Classification Examples <classification>`, :ref:`Regression Examples <regression>`