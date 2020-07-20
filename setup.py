from setuptools import setup, find_packages

setup(
    name='adaptation',
    version='0.1.0',
    description='Awesome Domain Adaptation Package Toolbox',
    url='https://github.com/antoinedemathelin/adapt.git',
    author='Antoine de Mathelin',
    author_email='antoine.de_mathelin@cmla.ens-cachan.fr',
    license='BSD-2',
    packages=find_packages(),
    install_requires=["numpy>=1.16", "scipy>=1.0", "tensorflow>=2.0", "scikit-learn>=0.2"],
    zip_safe=False
)
