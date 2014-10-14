# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from nose.tools import assert_raises

from .. import flow


def test_qn():
    """flow vectors q_n"""

    # This is _almost_ unnecessary.  It would be tough to screw up q_n.

    # q_n(0) = 1
    q = flow.qn(2, 0)
    assert q == 1., \
        'Incorrect single-particle q_n ({} != 1).'.format(q)

    # q_3(isotropic phi) = -1
    q = flow.qn(3, (0, np.pi/3, -np.pi/3))
    assert q == -1., \
        'Incorrect isotropic q_3 ({} != -1).'.format(q)


def test_flow_cumulant():
    """flow correlations and cumulants"""

    Nev = 10
    M = 100

    # Fabricate q_2 for perfectly fluctuating "flow" (no actual flow).
    qn = {2: M**.5*np.ones(Nev)}
    vnk = flow.FlowCumulant(M*np.ones(Nev), qn)

    corr22 = vnk.correlation(2, 2)
    assert corr22 == 0., \
        'Purely statistical correlation must vanish ({} != 0).'.format(corr22)

    c22 = vnk.cumulant(2, 2)
    assert c22 == 0., \
        'Purely statistical cumulant must vanish ({} != 0).'.format(c22)

    v22 = vnk.flow(2, 2)
    assert v22 == 0., \
        'Purely statistical flow must vanish ({} != 0).'.format(v22)

    assert_raises(ValueError, vnk.correlation, 2, 4)
