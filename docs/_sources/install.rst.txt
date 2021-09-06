.. _install:

Installation
============

Reference
---------

If you use this library in your research, please cite ADAPT using the following reference:

.. code-block::

	@article{de2021adapt,
	  title={ADAPT: Awesome Domain Adaptation Python Toolbox},
	  author={de Mathelin, Antoine and Deheeger, Fran{\c{c}}ois and Richard, Guillaume and Mougeot, Mathilde and Vayatis, Nicolas},
	  journal={arXiv preprint arXiv:2107.03049},
	  year={2021}
	}

Pypi Installation
-----------------

This package is available on `Pypi <https://badge.fury.io/py/adapt>`_. It has been tested on Linux, MacOSX and Windows
for Python versions: 3.6, 3.7 and 3.8. It can be installed with the following command line:

.. code-block:: python
	
	pip install adapt

The following dependencies are required and will be installed with the library:

- numpy
- scipy
- tensorflow (>= 2.0)
- scikit-learn
- cvxopt

If for some reason, these packages failed to install, you can do it manually with:

.. code-block:: python

	pip install numpy scipy tensorflow scikit-learn cvxopt

Finally import the module in your python scripts with:

.. code-block:: python

	import adapt