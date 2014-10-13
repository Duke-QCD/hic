# -*- coding: utf-8 -*-

from __future__ import division

import numexpr as ne


def qn(n, phi):
    return ne.evaluate('sum(exp(1j*n*phi))')
