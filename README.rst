hic
===

.. image:: http://img.shields.io/travis/Duke-QCD/hic.svg?style=flat-square
  :target: https://travis-ci.org/Duke-QCD/hic

.. image:: http://img.shields.io/coveralls/Duke-QCD/hic.svg?style=flat-square
  :target: https://coveralls.io/r/Duke-QCD/hic

Tools for analyzing heavy-ion collision simulations in Python.

Documentation
-------------
`qcd.phy.duke.edu/hic <http://qcd.phy.duke.edu/hic>`_

Installation
------------
Requirements: Python 2.7 or 3.3+ with numpy_.

``hic`` is prerelease software.
You can install the development version with pip_::

   pip install git+https://github.com/Duke-QCD/hic.git

To run the tests, install nose_ and run ::

   nosetests -v hic

Simple examples
---------------
Calculate flow cumulants:

.. code:: python

   from hic import flow

   vnk = flow.Cumulant(mult, q2, q3)
   v22 = vnk.flow(2, 2)

Randomly sample events with specified flows:

.. code:: python

   sampler = flow.Sampler(v2, v3)
   phi = sampler.sample(mult)

Calculate initial condition eccentricities:

.. code:: python

   from hic import initial

   ic = initial.IC(profile, dxy)
   e2 = ic.ecc(2)

.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _nose: https://nose.readthedocs.org
