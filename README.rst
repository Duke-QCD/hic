===
hic
===

Tools for analyzing heavy-ion collision simulations in Python.


Simple examples
---------------

.. code:: python

  from hic import flow

  vnk = flow.FlowCumulant(mult, ((2, q2), (3, q3)))
  v22 = vnk.flow(2, 2)
