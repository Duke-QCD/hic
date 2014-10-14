# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import numexpr as ne

__all__ = 'qn', 'FlowCumulant'


# If a variable is only ever used by numexpr, flake8 will flag it as unused.
# The comment 'noqa' prevents this warning.


def qn(n, phi):
    return ne.evaluate('sum(exp(1j*n*phi))')


class FlowCumulant(object):
    def __init__(self, multiplicities, qn):
        self.multiplicities = np.asarray(multiplicities)
        self._qn = dict(qn)
        self._corr2 = {}
        self._corr4 = {}

    def _calculate_corr2(self, n):
        try:
            qn = self._qn[n]  # noqa
        except KeyError:
            raise

        M = self.multiplicities  # noqa
        self._corr[n][2] = ne.evaluate(
            'sum(qn*conj(qn) - M) / sum(M*(M-1))'
        )

    def _calculate_corr4(self, n):
        pass

    def _get_corr(self, n, k):
        pass

    def correlation(self, n, k):
        pass

    def cumulant(self, n, k, error=False, negative_imaginary=False):
        pass
