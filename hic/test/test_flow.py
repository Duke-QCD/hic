# -*- coding: utf-8 -*-

from __future__ import division

import itertools
import warnings

import numpy as np
from nose.tools import assert_raises

from .. import flow


def test_qn():
    """flow vectors q_n"""

    # This is _almost_ unnecessary.  It would be tough to screw up q_n.

    # q_n(0) = 1
    q = flow.qn(2, 0)
    assert q == 1., \
        'Incorrect single-particle q_n.\n{} != 1'.format(q)

    # q_3(isotropic phi) = -1
    q = flow.qn(3, (0, np.pi/3, -np.pi/3))
    assert q == -1., \
        'Incorrect isotropic q_3.\n{} != -1'.format(q)


def _check_phi(M, *args):
    phi = flow.phi_event(M, *args)

    assert phi.size == M, \
        'Incorrect number of particles.'
    assert np.all((phi >= -np.pi) & (phi < np.pi)), \
        'Azimuthal angle not in [-pi, pi).'


def test_phi_event():
    """event generation"""

    M = 10
    _check_phi(M)
    _check_phi(M, .1)
    _check_phi(M, (.1, 0, .01))
    _check_phi(M, (.1, 0, .01), 1.)
    _check_phi(M, (.1, 0, .01), (1., 0, 1.2))


def assert_close(a, b, msg='', tol=1e-15):
    assert abs(a - b) < tol, \
        '{}\n{} != {}'.format(msg, a, b)


def test_flow_cumulant():
    """flow correlations and cumulants"""

    # Compare n-particle correlations to explicitly calculating all
    # n-permutations.
    M = 8
    phi = 2*np.pi*np.random.rand(8)

    corr2_true = np.exp(2j*np.array([
        (p1-p2) for p1, p2 in itertools.permutations(phi, 2)
    ])).mean()
    corr4_true = np.exp(2j*np.array([
        (p1+p2-p3-p4) for p1, p2, p3, p4 in itertools.permutations(phi, 4)
    ])).mean()

    qn = {n: np.exp(1j*n*phi).sum() for n in (2, 4)}
    vnk = flow.FlowCumulant(M, qn)
    corr2, corr4 = (vnk.correlation(2, n) for n in (2, 4))

    assert_close(corr2, corr2_true, 'Incorrect 2-particle correlation.')
    assert_close(corr4, corr4_true, 'Incorrect 4-particle correlation.')

    # Now verify cumulants.
    c2_true = corr2_true
    c4_true = corr4_true - 2*corr2_true*corr2_true

    c2, c4 = (vnk.cumulant(2, n) for n in (2, 4))

    assert_close(c2, c2_true, 'Incorrect 2-particle cumulant.')
    assert_close(c4, c4_true, 'Incorrect 4-particle cumulant.')

    # Fabricate q_n so that v2 is positive, v3 vanishes, and v4 is imaginary.
    Nev = 10
    M = 100
    qfluct = M**.5*np.ones(Nev)
    qn = {2: 1.1*qfluct, 3: qfluct, 4: 0.9*qfluct}
    vnk = flow.FlowCumulant(M*np.ones(Nev), qn)

    v22 = vnk.flow(2, 2)
    assert v22 > 0, \
        'This v_2{2} must be positive.\n{}'.format(v22)

    v32 = vnk.flow(3, 2)
    assert v32 == 0, \
        'This v_3{2} must vanish.\n{}'.format(v32)

    with warnings.catch_warnings(record=True) as w:
        v42_nan = vnk.flow(4, 2)
        assert len(w) == 1 and issubclass(w[0].category, RuntimeWarning), \
            'There should be RuntimeWarning here.'
    assert np.isnan(v42_nan), \
        'This v_4{2} must be NaN.\n{}'.format(v42_nan)

    v42_zero = vnk.flow(4, 2, imaginary='zero')
    assert v42_zero == 0, \
        'This v_4{2} must be zero.\n{}'.format(v42_zero)

    v42_negative = vnk.flow(4, 2, imaginary='negative')
    assert v42_negative < 0, \
        'This v_4{2} must be negative.\n{}'.format(v42_negative)

    # Test bad arguments.

    # invalid k
    assert_raises(ValueError, vnk.correlation, 2, 3)

    # missing qn
    assert_raises(ValueError, vnk.cumulant, 4, 4)
