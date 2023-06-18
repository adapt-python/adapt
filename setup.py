from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='adapt',
    version='0.4.3',
    description='Awesome Domain Adaptation Python Toolbox for Tensorflow and Scikit-learn',
    url='https://github.com/adapt-python/adapt.git',
    author='Antoine de Mathelin',
    author_email='antoine.demat@gmail.com',
    license='BSD-2',
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy>=1.16", "scipy>=1.0", "tensorflow<2.12", "scikit-learn>=0.2", "cvxopt<=1.3.0"],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
