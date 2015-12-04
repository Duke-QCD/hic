#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import hic

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='hic',
    version=hic.__version__,
    description='Tools for analyzing heavy-ion collision simulations.',
    long_description=long_description,
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/Duke-QCD/hic',
    license='MIT',
    packages=['hic', 'hic.test'],
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
