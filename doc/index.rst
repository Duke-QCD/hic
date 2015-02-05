hic
===

``hic`` is a collection of Python modules for analyzing heavy-ion collision simulation data,
e.g. calculating flow coefficients `v_n` and initial condition eccentricities `\varepsilon_n`.
It's a community project designed to facilitate computational heavy-ion research and reduce code duplication.

``hic`` is under active development on github_, and :ref:`contributions <contributing>` are welcome!


Installation
------------
Requirements: Python 2.7 or 3.3+ with numpy_.

``hic`` is prerelease software.
You can install the development version with pip_::

   pip install git+https://github.com/jbernhard/hic#egg=hic

Or, if you want to edit the code, clone the repository and install in editable mode::

   git clone git@github.com:jbernhard/hic.git
   pip install [--user] -e hic

To run the tests, install nose_ and run ::

   nosetests -v hic

Simple examples
---------------
Here are a few quick examples of what ``hic`` can do.

Calculate flow cumulants::

  from hic import flow

  vnk = flow.FlowCumulant(mult, {2: q2, 3: q3})
  v22 = vnk.flow(2, 2)

Randomly sample events with specified flows::

  phi = flow.sample_flow_pdf(mult, vn=(v2, v3))

Calculate initial condition eccentricities::

  from hic import initial

  ic = initial.IC(profile, dxy)
  e2 = ic.ecc(2)

User guide
----------
``hic`` consists of several logically distinct modules.
Each has a tutorial with examples and an API reference.

.. toctree::
   :maxdepth: 2

   flow
   initial

.. _contributing:

Contributing
------------
``hic`` is a community project---all heavy-ion physicists are encouraged to contribute!
Anything is welcome, from a small code snippet to a completely new module to an extra example for the docs.
Here's a short wish list:

- flow cumulant statistical error
- differential flow
- input/output for common formats (e.g. UrQMD)
- an HBT module
- a heavy-flavor module

All code should be :PEP:`8`-compliant (check with flake8_) and have high-coverage unit tests (using nose_) and documentation (for sphinx_).
See existing code for examples of how to write tests and docs.

Submit contributions through a github `pull request`_.

.. _github: https://github.com/jbernhard/hic
.. _issue: https://github.com/jbernhard/hic/issues
.. _pull request: https://github.com/jbernhard/hic/pulls
.. _numpy: http://www.numpy.org
.. _pip: https://pip.pypa.io
.. _nose: https://nose.readthedocs.org
.. _flake8: http://flake8.readthedocs.org
.. _nose: https://nose.readthedocs.org
.. _sphinx: http://sphinx-doc.org
