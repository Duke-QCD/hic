===
hic
===

Tools for analyzing heavy-ion collision simulations in Python.

.. image:: https://travis-ci.org/jbernhard/hic.svg?branch=master
  :target: https://travis-ci.org/jbernhard/hic

.. image:: https://coveralls.io/repos/jbernhard/hic/badge.png?branch=master
  :target: https://coveralls.io/r/jbernhard/hic?branch=master

Simple examples
---------------

.. code:: python

  from hic import flow

  vnk = flow.FlowCumulant(mult, {2: q2, 3: q3})
  v22 = vnk.flow(2, 2)
