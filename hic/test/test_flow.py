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
    q = flow.qn(0, 2)
    assert q == 1., \
        'Incorrect single-particle q_n.\n{} != 1'.format(q)

    # q_3(isotropic phi) = -1
    q = flow.qn((0, np.pi/3, -np.pi/3), 3)
    assert q == -1., \
        'Incorrect isotropic q_3.\n{} != -1'.format(q)

    n_ = 2, 3
    phi = np.random.uniform(-np.pi, np.pi, 10)
    assert np.allclose(
        flow.qn(phi, *n_),
        [flow.qn(phi, n) for n in n_]
    ), 'Simultaneous qn do not agree with individuals.'


def assert_close(a, b, msg='', tol=1e-15):
    assert abs(a - b) < tol, \
        '{}\n{} != {}'.format(msg, a, b)


def test_cumulant():
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

    q2, q4 = (np.exp(1j*n*phi).sum() for n in (2, 4))
    vnk = flow.Cumulant(M, q2, None, q4)
    assert sorted(vnk._qn.keys()) == [2, 4], \
        'Incorrect parsing of qn arguments.'
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
    qn = 1.1*qfluct, qfluct, 0.9*qfluct
    vnk = flow.Cumulant(M*np.ones(Nev), *qn)

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

    # Test statistical error.
    v22, v22err = vnk.flow(2, 2, error=True)
    assert v22err < 1e-16, \
        'This v_2{2} error must vanish.\n{}'.format(v22err)
    assert (v22, v22err) == vnk.flow(2, 2, error=True), \
        'Cached result does not match original.'

    Nev = 5
    M = 10
    w1 = Nev*M*(M-1)
    w2 = Nev*np.square(M*(M-1))

    phi_events = 2*np.pi*np.random.rand(Nev, M)
    q2, q4 = (np.array([flow.qn(phi, n) for phi in phi_events])
              for n in (2, 4))
    vnk = flow.Cumulant(np.full(Nev, M), q2=q2, q4=q4)

    for n in 2, 4:
        corr_ebe = np.array([
            np.exp(n*1j*np.array([
                (p1-p2) for p1, p2 in itertools.permutations(phi, 2)
            ])).mean()
            for phi in phi_events])

        corr_true = corr_ebe.mean()
        corr_err_true = np.sqrt(w2/(w1*w1 - w2)) * corr_ebe.std()

        corr, corr_err = vnk.correlation(n, 2, error=True)
        assert_close(corr, corr_true,
                     'Incorrect {}-particle correlation.'.format(n))
        assert_close(corr_err, corr_err_true,
                     'Incorrect {}-particle correlation error.'.format(n))

    # Test bad arguments.

    # invalid k
    assert_raises(ValueError, vnk.correlation, 2, 3)

    # error only implemented for k == 2
    assert_raises(ValueError, vnk.correlation, 2, 4, error=True)

    # missing qn
    assert_raises(ValueError, vnk.cumulant, 4, 4)

    # bad class keyword args
    assert_raises(TypeError, flow.Cumulant, M, v2=q2)


def test_sampler():
    """flow random sampling"""

    # zero flow
    sampler = flow.Sampler()
    assert sampler._n is None and sampler._vn is None, \
        'Incorrect vn argument parsing.'

    phi = np.random.rand(5)
    assert np.allclose(sampler.pdf(phi), 1/(2*np.pi)), \
        'Incorrect uniform flow pdf.'

    M = 1000
    phi = sampler.sample(M)
    vnobs = np.cos(np.outer(np.arange(2, 6), phi)).mean(axis=1)
    assert phi.size == M, \
        'Incorrect number of particles.'
    assert np.all((phi >= -np.pi) & (phi < np.pi)), \
        'Sampled angles are not in [-pi, pi).'
    assert np.all(np.abs(vnobs) < 3.*M**(-.5)), \
        'Flows are not within statistical fluctuation.'

    # nonzero flow
    vn = v2, v3, v4, v5 = .1, .05, 0., .02
    for sampler in (
            flow.Sampler(v2, v3, None, v5),
            flow.Sampler(v2, v3, v5=v5),
            flow.Sampler(v5=v5, v2=v2, v3=v3)
    ):
        assert (
            np.all(sampler._n == (2, 3, 5)) and
            np.all(sampler._vn == (v2, v3, v5))
        ), 'Incorrect vn argument parsing.'

    pdf = (1 +
           2*v2*np.cos(2*phi) +
           2*v3*np.cos(3*phi) +
           2*v5*np.cos(5*phi)
           )/(2*np.pi)
    assert np.allclose(sampler.pdf(phi), pdf), \
        'Incorrect nonuniform flow pdf.'

    M = 1000
    phi = sampler.sample(M)
    vnobs = np.cos(np.outer(np.arange(2, 6), phi)).mean(axis=1)
    assert phi.size == M, \
        'Incorrect number of particles.'
    assert np.all((phi >= -np.pi) & (phi < np.pi)), \
        'Sampled angles are not in [-pi, pi).'
    assert np.all(np.abs(vn - vnobs) < 3.*M**(-.5)), \
        'Flows are not within statistical fluctuation.'

    # bad args
    assert_raises(TypeError, flow.Sampler, M, x2=.1)
