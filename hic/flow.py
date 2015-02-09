# -*- coding: utf-8 -*-

from __future__ import division

import collections
import warnings

import numpy as np

__all__ = 'qn', 'flow_pdf', 'sample_flow_pdf', 'FlowCumulant'


def qn(phi, *n):
    """
    Calculate the complex flow vector `Q_n`.

    :param array-like phi: Azimuthal angles.

    :param int n: One or more harmonics to calculate.

    :returns:
        A single complex number if only one ``n`` was given or a complex array
        for multiple ``n``.

    """
    phi = np.ravel(phi)
    n = np.asarray(n)

    i_n_phi = np.zeros((n.size, phi.size), dtype=complex)
    np.outer(n, phi, out=i_n_phi.imag)

    qn = np.exp(i_n_phi, out=i_n_phi).sum(axis=1)
    if qn.size == 1:
        qn = qn[0]

    return qn


def _uniform_phi(M):
    """
    Generate M random numbers in [-pi, pi).

    """
    return np.random.uniform(-np.pi, np.pi, M)


def _flow_pdf_unnormalized(phi, n, vn):
    """
    Evaluate the unnormalized flow PDF at phi.
    n is an array of flow orders.
    vn is an array of the corresponding flows.

    """
    pdf = np.inner(vn, np.cos(np.outer(phi, n)))
    pdf *= 2.
    pdf += 1.
    return pdf


def _parse_vn(vn, other_vn):
    """
    Parse a list of vn and a dict of vn (as input to flow_pdf and
    sample_flow_pdf) into two arrays suitable for input to
    _flow_pdf_unnormalized.

    """
    vn_dict = {int(k.lstrip('v')): v for k, v in other_vn.items()}
    vn_dict.update((k, v) for k, v in enumerate(vn, start=2)
                   if v is not None and v != 0.)
    kwargs = dict(dtype=float, count=len(vn_dict))
    n = np.fromiter(vn_dict.keys(), **kwargs)
    vn = np.fromiter(vn_dict.values(), **kwargs)
    return n, vn


def flow_pdf(phi, *vn, **other_vn):
    r"""
    Evaluate the flow probability density function `dN/d\phi`.

    :param array-like phi: Azimuthal angles.

    :param float v2, v3, ...:
        Flow coefficients as positional arguments.

    :param float other_vn:
        Flow coefficients as keyword arguments.

    :returns: The flow PDF evaluated at ``phi``.

    """
    if not vn and not other_vn:
        pdf = np.empty_like(phi)
        pdf.fill(.5/np.pi)
        return pdf

    phi = np.asarray(phi)
    n, vn = _parse_vn(vn, other_vn)

    pdf = _flow_pdf_unnormalized(phi, n, vn)
    pdf /= 2.*np.pi

    return pdf


def sample_flow_pdf(multiplicity, *vn, **other_vn):
    r"""
    Randomly sample azimuthal angles `\phi` with specified flows.

    To sample uniform `\phi`, do not specify any flows.

    :param int multiplicity: Number to sample.

    :param float v2, v3, ...:
        Flow coefficients as positional arguments.

    :param float other_vn:
        Flow coefficients as keyword arguments.

    :returns: Array of sampled angles.

    """
    if not vn and not other_vn:
        return _uniform_phi(multiplicity)

    n, vn = _parse_vn(vn, other_vn)

    # Since the flow PDF does not have an analytic inverse CDF, I use a simple
    # accept-reject sampling algorithm.  This is reasonably efficient since for
    # normal-sized vn, the PDF is close to flat.  Now due to the overhead of
    # Python functions, it's desirable to minimize the number of calls to the
    # random number generator.  Therefore I sample numbers in chunks; most of
    # the time only one or two chunks should be needed.  Eventually, I might
    # rewrite this with Cython, but it's fast enough for now.

    N = 0  # number of phi that have been sampled
    phi = np.empty(multiplicity)  # allocate array for phi
    pdf_max = 1 + 2*vn.sum()  # sampling efficiency ~ 1/pdf_max
    while N < multiplicity:
        n_remaining = multiplicity - N
        n_to_sample = int(1.03*pdf_max*n_remaining)
        phi_chunk = _uniform_phi(n_to_sample)
        phi_chunk = phi_chunk[(_flow_pdf_unnormalized(phi_chunk, n, vn) >
                               np.random.uniform(0, pdf_max, n_to_sample))]
        K = min(phi_chunk.size, n_remaining)  # number of phi to take
        phi[N:N+K] = phi_chunk[:K]
        N += K

    return phi


class FlowCumulant(object):
    r"""
    Multi-particle flow correlations and cumulants for an ensemble of events.

    Each argument must be an array-like object where each element corresponds
    to an event: ``multiplicities`` is an array containing event-by-event
    multiplicities, ``q2`` is an array of the same size containing the `Q_2`
    vectors for the same set of events, and so on.

    :param array-like multiplicities: Event-by-event multiplicities.

    :param array-like q2, q3, ...:
        `Q_n` vectors as positional arguments.

    :param array-like other_qn:
        `Q_n` vectors as keyword arguments.

    """
    def __init__(self, multiplicities, *qn, **other_qn):
        # Multiplicity must be stored as floating point because the large
        # powers of M calculated in n-particle correlations can overflow
        # integers, e.g. 2000^6 > 2^64.
        self._M = np.asarray(multiplicities, dtype=float).ravel()

        qn_dict = {int(k.lstrip('q')): v for k, v in other_qn.items()}
        qn_dict.update(enumerate(qn, start=2))
        self._qn = {k: np.asarray(v, dtype=complex).ravel()
                    for k, v in qn_dict.items()
                    if v is not None}

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

        # TODO: put these functions in a _util module
        M = self._M
        Msum = np.einsum('i->', M)  # fast sum(M)
        Msqsum = np.inner(M, M)  # fast sum(M^2)

        qn = self._get_qn(n)
        qnsqsum = np.vdot(qn, qn).real  # fast sum(|qn|^2)

        if k == 2:
            self._corr[n][k] = (qnsqsum - Msum) / (Msqsum - Msum)

        elif k == 4:
            q2n = self._get_qn(2*n)
            q2nsqsum = np.vdot(q2n, q2n).real
            qnsq = np.square(qn.real) + np.square(qn.imag)
            qnto4sum = np.inner(qnsq, qnsq)
            self._corr[n][k] = (
                qnto4sum +
                q2nsqsum -
                2*np.inner(q2n, np.square(qn.conj())).real -
                2*np.sum(2*(M-2)*qnsq - M*(M-3))
            ) / np.sum(M*(M-1)*(M-2)*(M-3))

        # TODO: k == 6

        else:
            raise ValueError('Unknown k: {}.'.format(k))

    def correlation(self, n, k):
        r"""
        Calculate `\langle k \rangle_n`,
        the `k`-particle correlation function for `n`\ th-order anisotropy.

        :param int n: Flow order.
        :param int k: Correlation order.

        """
        self._calculate_corr(n, k)
        return self._corr[n][k]

    def cumulant(self, n, k, error=False):
        r"""
        Calculate `c_n\{k\}`,
        the `k`-particle cumulant for `n`\ th-order anisotropy.

        :param int n: Flow order.
        :param int k: Correlation order.

        """
        corr_nk = self.correlation(n, k)

        if k == 2:
            return corr_nk
        elif k == 4:
            corr_n2 = self.correlation(n, 2)
            return corr_nk - 2*corr_n2*corr_n2

    _cnk_prefactor = {2: 1, 4: -1}

    def flow(self, n, k, error=False, imaginary='nan'):
        r"""
        Calculate `v_n\{k\}`,
        the estimate of flow coefficient `v_n` from the `k`-particle cumulant.

        :param int n: Flow order.
        :param int k: Correlation order.

        :param str imaginary: (optional)
            Determines behavior when the computed flow is imaginary:

            - ``'nan'`` (default) -- Return NaN and raise a ``RuntimeWarning``.
            - ``'negative'`` -- Return the negative absolute value.
            - ``'zero'`` -- Return ``0.0``.

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
