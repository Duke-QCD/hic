# -*- coding: utf-8 -*-

from __future__ import division

import collections
import warnings

import numpy as np
import numexpr as ne

__all__ = 'qn', 'FlowCumulant'


# If a variable is only ever used by numexpr, flake8 will flag it as unused.
# The comment 'noqa' prevents this warning.


def qn(n, phi):
    return ne.evaluate('sum(exp(1j*n*phi))')


class FlowCumulant(object):
    def __init__(self, multiplicities, qn):
        self._M = np.asarray(multiplicities, dtype=np.float64)
        self._qn = {n: np.asarray(q, dtype=np.complex128)
                    for n, q in dict(qn).items()}
        self._corr = collections.defaultdict(dict)

    def _get_qn(self, n):
        try:
            return self._qn[n]
        except KeyError:
            raise ValueError(
                'q_{} is required for this calculation but was not provided.'
                .format(n)
            )

    def _calculate_corr(self, n, k):
        if k in self._corr[n]:
            return

        M = self._M  # noqa
        qn = self._get_qn(n)  # noqa

        if k == 2:
            numerator = ne.evaluate('sum(real(qn*conj(qn)) - M)')
            denominator = ne.evaluate('sum(M*(M-1))')

        elif k == 4:
            q2n = self._get_qn(2*n)  # noqa
            numerator = ne.evaluate(
                '''sum(
                real(qn*conj(qn))**2 +
                real(q2n*conj(q2n)) -
                2*real(q2n*conj(qn)*conj(qn)) -
                4*(M-2)*real(qn*conj(qn)) +
                2*M*(M-3)
                )'''
            )
            denominator = ne.evaluate('sum(M*(M-1)*(M-2)*(M-3))')

        else:
            raise ValueError('Unknown k: {}.'.format(k))

        self._corr[n][k] = numerator / denominator

    def correlation(self, n, k):
        self._calculate_corr(n, k)
        return self._corr[n][k]

    def cumulant(self, n, k, error=False):
        corr_nk = self.correlation(n, k)

        if k == 2:
            return corr_nk
        elif k == 4:
            corr_n2 = self.correlation(n, 2)
            return corr_nk - 2*corr_n2*corr_n2

    _cnk_prefactor = {2: 1, 4: -1}

    def flow(self, n, k, error=False, imaginary='nan'):
        cnk = self.cumulant(n, k)
        vnk_to_k = self._cnk_prefactor[k] * cnk
        kinv = 1/k

        if vnk_to_k >= 0:
            vnk = vnk_to_k**kinv
        else:
            if imaginary == 'negative':
                vnk = -1*(-vnk_to_k)**kinv
            elif imaginary == 'zero':
                vnk = 0.
            else:
                warnings.warn('Imaginary flow: returning NaN.', RuntimeWarning)
                vnk = float('nan')

        return vnk
