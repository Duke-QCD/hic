# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .. import flow


def test_qn(seed=1248):
    # q_n(0) = 1
    q = flow.qn(2, 0)
    assert q == 1+0j, \
        'Incorrect single-particle q_n ({} != 1).'.format(q)

    # q_3(uniform phi) = -1
    q = flow.qn(3, np.arange(-np.pi, np.pi, 10))
    assert abs(q+1) < 1e-12, \
        'Incorrect isotropic q_n ({} != -1).'.format(q)

    # specific example
    np.random.seed(seed)
    phi = 2*np.pi*(np.random.rand(10) - .5)
    q = np.array([flow.qn(n, phi) for n in range(2, 5)])
    correct_q = np.array((
        -0.23701789876111995+1.9307467860155012j,
        0.7294873796006498+0.4925428484240118j,
        2.0248053489550459-0.23452484252744438j
    ))
    assert np.allclose(q, correct_q), \
        'Incorrect random q_n.\n{} != {}'.format(q, correct_q)


def test_flow_cumulant(seed=1248):
    np.random.seed(seed)
    Nev = 10
    M = 10
    events = 2*np.pi*(np.random.rand(Nev, M) - .5)
    qn = (
        (n, np.array([flow.qn(n, phi) for phi in events]))
        for n in range(2, 5)
    )
    vnk = flow.FlowCumulant(M*np.ones(Nev), qn)
    v22 = vnk.cumulant(2, 2)
