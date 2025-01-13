========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |commits-since|


.. |github-actions| image:: https://github.com/rattinge/surfwetter-ml/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/rattinge/surfwetter-ml/actions

.. |codecov| image:: https://codecov.io/gh/rattinge/surfwetter-ml/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/rattinge/surfwetter-ml

.. |commits-since| image:: https://img.shields.io/github/commits-since/rattinge/surfwetter-ml/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/rattinge/surfwetter-ml/compare/v0.1.0...main



.. end-badges

ML wind forecast to find the best lake for surfing in Switzerland and abroad

* Free software: MIT license

Installation
============

::

    pip install surfwetter-ml

You can also install the in-development version with::

    pip install https://github.com/rattinge/surfwetter-ml/archive/main.zip


Documentation
=============


https://docs.surfwetter.ch


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
