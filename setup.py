import os
from setuptools import setup, find_packages

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='adapt',
    version='0.4.5',
    description='Awesome Domain Adaptation Python Toolbox for Tensorflow and Scikit-learn',
    url='https://github.com/adapt-python/adapt.git',
    author='Antoine de Mathelin',
    author_email='antoine.demat@gmail.com',
    license='BSD-2',
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy", "scipy", "tensorflow", "scikit-learn", "cvxopt", "scikeras"],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
