# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .. import flow


def test_qn():
    assert flow.qn(2, 0) == 1+0j, \
        'Single-particle q_n.'

    assert np.allclose(flow.qn(3, np.arange(-np.pi, np.pi, 10)), -1+0j), \
        'Isotropic q_n.'


def test_flow_cumulant():
    pass
