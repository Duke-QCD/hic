# -*- coding: utf-8 -*-

from __future__ import division

import collections
import warnings

import numpy as np

__all__ = 'qn', 'FlowCumulant'


def qn(n, phi):
    phi = np.asarray(phi)
    return np.exp(1j*n*phi).sum()


def square_complex(z):
    return np.square(z.real) + np.square(z.imag)


class FlowCumulant(object):
    """
    Flow correlation and cumulant calculator.

    multiplicities: (nevents,)
        Event-by-event multiplicities.

    qn:
        Event-by-event q_n vectors.  Must be an object that can be directly
        used to construct a dict of the form {n: q_n, m: q_m, ...}.

    """
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
        # skip if this correlation was already calculated
        if k in self._corr[n]:
            return

        M = self._M
        qn = self._get_qn(n)

        if k == 2:
            self._corr[n][k] = np.sum(square_complex(qn) - M) / np.sum(M*(M-1))

        elif k == 4:
            q2n = self._get_qn(2*n)
            qnsq = square_complex(qn)
            self._corr[n][k] = (
                np.sum(
                    np.square(qnsq) +
                    square_complex(q2n) -
                    2*(q2n*np.square(qn.conj())).real -
                    4*(M-2)*qnsq +
                    2*M*(M-3)
                ) / np.sum(M*(M-1)*(M-2)*(M-3))
            )

        else:
            raise ValueError('Unknown k: {}.'.format(k))

    def correlation(self, n, k):
        """
        Calculate k-particle correlation for nth-order anisotropy.

        """
        self._calculate_corr(n, k)
        return self._corr[n][k]

    def cumulant(self, n, k, error=False):
        """
        Calculate cumulant c_n{k}.

        """
        corr_nk = self.correlation(n, k)

        if k == 2:
            return corr_nk
        elif k == 4:
            corr_n2 = self.correlation(n, 2)
            return corr_nk - 2*corr_n2*corr_n2

    _cnk_prefactor = {2: 1, 4: -1}

    def flow(self, n, k, error=False, imaginary='nan'):
        """
        Calculate flow v_n{k}.

        imaginary (optional):
            Determines what is returned when the flow is imaginary:
            * 'nan' (default) -> NaN, and raise RuntimeWarning
            * 'negative' -> negative absolute value
            * 'zero' -> 0.0

        """
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
