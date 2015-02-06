hic
===

.. image:: http://img.shields.io/travis/jbernhard/hic.svg?style=flat-square
  :target: https://travis-ci.org/jbernhard/hic

.. image:: http://img.shields.io/coveralls/jbernhard/hic.svg?style=flat-square
  :target: https://coveralls.io/r/jbernhard/hic

Tools for analyzing heavy-ion collision simulations in Python.

Documentation
-------------
`jbernhard.github.io/hic <http://jbernhard.github.io/hic>`_

Installation
------------
Requirements: Python 2.7 or 3.3+ with numpy_.

``hic`` is prerelease software.
You can install the development version with pip_::

   pip install git+https://github.com/jbernhard/hic.git

To run the tests, install nose_ and run ::

   nosetests -v hic

Simple examples
---------------
Calculate flow cumulants:

.. code:: python

  from hic import flow

  vnk = flow.FlowCumulant(mult, {2: q2, 3: q3})
  v22 = vnk.flow(2, 2)

Randomly sample events with specified flows:

.. code:: python

  phi = flow.sample_flow_pdf(mult, vn=(v2, v3))

Calculate initial condition eccentricities:

.. code:: python

  from hic import initial

  ic = initial.IC(profile, dxy)
  e2 = ic.ecc(2)

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _nose: https://nose.readthedocs.org
