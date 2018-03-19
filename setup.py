#! /usr/bin/env python
#
# Copyright (C) 2016 Russell Poldrack <poldrack@stanford.edu>
# some portions borrowed from https://github.com/mwaskom/lyman/blob/master/setup.py
from setuptools import setup,find_packages

descr="""Probabilistic Context Task"""
DISTNAME="Prob_Context_Task"
DESCRIPTION=descr
MAINTAINER='Ian Eisenberg'
MAINTAINER_EMAIL='ieisenbe@stanford.edu'
LICENSE='MIT'
DOWNLOAD_URL='https://github.com/IanEisenberg/Prob_Context_Task'
VERSION='0.1.0.dev'

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    classifiers=['Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6.4',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS'],
)
