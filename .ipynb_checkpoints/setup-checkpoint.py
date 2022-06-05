import os
from setuptools import setup

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# read the docs could not compile numpy and c extensions
if on_rtd:
    setup_requires = []
    install_requires = []
else:
    setup_requires = [
        'nose',
        'coverage',
    ]
    install_requires = [
        'six',
        'numpy',
        'scipy',
        'matplotlib',
    ]

long_description = ("See `github <https://github.com/leepand/onlineLearning>`_ "
                    "for more information.")

setup(
    name='onlineLearning',
    version='0.0.1',
    description='Contextual bandit in python',
    long_description=long_description,
    author='leepand',
    author_email='pandeng.li@163.com',
    url='https://github.com/leepand/onlineLearning',
    setup_requires=setup_requires,
    install_requires=install_requires,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='nose.collector',
    packages=[
        'onlineLearning',
        'onlineLearning.learners',
        'onlineLearning.pipes',
        'onlineLearning.storage'
    ],
    package_dir={
        'onlineLearning': 'onlineLearning',
        'onlineLearning.learners': 'onlineLearning/learners',
        'onlineLearning.pipes': 'onlineLearning/pipes',
        'onlineLearning.storage': 'onlineLearning/storage'
    },
)