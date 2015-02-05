Anisotropic flow
================
The ``flow`` module handles anisotropic flow coefficients `v_n` and related tasks.

Flows `v_n` are the Fourier coefficients of the transverse momentum distribution:

.. math::

   \frac{dN}{d\phi} = \frac{1}{2\pi} \biggl[ 1 + \sum_{n=1}^\infty 2v_n\cos n(\phi - \Psi_\text{RP}) \biggr],

where `\phi` is the azimuthal angle of transverse momentum, `n` is the order of anisotropy, and `\Psi_\text{RP}` is the reaction plane angle.
The coefficients are

.. math::

   v_n = \langle \cos n(\phi - \Psi_\text{RP}) \rangle

where the average is over all particles and events.

The flow vector
---------------
The complex flow vector for an event is

.. math::
   
   Q_n = \sum_{i=1}^M e^{in\phi_i}.

Flow vectors can be calculated in ``hic`` with ``hic.flow.qn``.
First, let's generate some random angles::

   import numpy as np
   phi = np.random.uniform(-np.pi, np.pi, 100)

And now calculate `Q_2`::

   from hic import flow
   q2 = flow.qn(2, phi)

We can also do multiple `Q_n` at once::

   q2, q3, q4 = flow.qn((2, 3, 4), phi)

Flow cumulants
--------------
Multi-particle flow cumulants are implemented in the class ``hic.flow.FlowCumulant`` using the direct *Q*-cumulant method from `Bilandzic (2010) <http://inspirehep.net/record/871528>`_.

Random sampling
---------------

Reference
---------
.. automodule:: hic.flow
   :members:
