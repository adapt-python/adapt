.. _install:

Installation
============

ADAPT is a python library which provides several domain adaptation methods usefull to improve machine learning models.

This Python library provides several solvers for optimization problems related to Optimal Transport for signal, image processing and machine learning.

The library has been tested on Linux, MacOSX and Windows. It requires a C++ compiler for building/installing the EMD solver and relies on the following Python modules:

    Numpy (>=1.16)

    Scipy (>=1.0)

    Cython (>=0.23)

    Matplotlib (>=1.5)

Pip installation

Note that due to a limitation of pip, cython and numpy need to be installed prior to installing POT. This can be done easily with

pip install numpy cython

You can install the toolbox through PyPI with:

pip install POT

or get the very latest version by running:

pip install -U https://github.com/PythonOT/POT/archive/master.zip # with --user for user install (no root)

