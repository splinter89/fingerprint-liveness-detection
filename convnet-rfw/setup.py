from setuptools import setup, find_packages

setup(
    name='cnnrandom',
    version='0.1dev',
    packages=find_packages(),
    description='A minimal implementation of Convolutional Neural Networks '
                'using Random Filter Weights for Image Feature Extraction.',
    license='BSD 3-clause license',
    long_description=open('README.md').read(),
    install_requires=['numpy>=1.6.1',
                      'scipy>=0.10.0',
                      'numexpr>=2.0',
                      'scikit-image>=0.5',
                      'nose>=1.3.0'],
    package_data={'': ['*.md']},
)
