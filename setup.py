#!/usr/bin/env python
# coding=utf-8

import codecs

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Topic :: Utilities
Topic :: Scientific/Engineering
License :: OSI Approved :: Apache Software License
"""

try:
    from shapeguard import __about__
    about = __about__.__dict__
except ImportError:
    # installing - dependencies are not there yet
    # Manually extract the __about__
    about = dict()
    exec(open("shapeguard/__about__.py").read(), about)


setup(
    name='shapeguard',
    version=about['__version__'],

    author=about['__author__'],
    author_email=about['__author_email__'],

    url=about['__url__'],

    packages=['shapeguard'],

    classifiers=list(filter(None, classifiers.split('\n'))),
    description='ShapeGuard is a tool to help with handling shapes.',
    long_description=codecs.open('README.md', encoding='utf_8').read()
)
