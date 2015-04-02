# -*- coding: utf-8 -*-

from __future__ import division

import collections
import warnings

import numpy as np

__all__ = 'qn', 'Cumulant', 'Sampler'


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


class Cumulant(object):
    r"""
    Multi-particle flow correlations and cumulants for an ensemble of events.

    Each argument must be an array-like object where each element corresponds
    to an event: ``multiplicities`` is an array containing event-by-event
    multiplicities, ``q2`` is an array of the same size containing the `Q_2`
    vectors for the same set of events, and so on.

    :param array-like multiplicities: Event-by-event multiplicities.

    :param array-like q2, q3, ...:
        `Q_n` vectors as positional arguments.

    :param array-like qn_kwargs:
        `Q_n` vectors as keyword arguments.

    Member functions ``correlation``, ``cumulant``, and ``flow`` compute the
    correlation function, cumulant, and flow coefficient, respectively.  Each
    takes two arguments ``(n, k)`` where ``n`` (positive integer) is the
    anisotropy order and ``k`` (even positive integer) is the correlation
    order.  A given `(n, k)` requires flow vectors `Q_n, Q_{2n}, \ldots,
    Q_{nk/2}`, e.g. ``(2, 4)`` requires ``q2, q4``.  Functions will raise
    ``ValueError`` if they don't have the required flow vectors.  Currently
    ``k=2`` and ``k=4`` are implemented; ``k=6`` is planned.

    """
    def __init__(self, multiplicities, *qn, **qn_kwargs):
        # Multiplicity must be stored as floating point because the large
        # powers of M calculated in n-particle correlations can overflow
        # integers, e.g. 2000^6 > 2^64.
        self._M = np.asarray(multiplicities, dtype=float).ravel()

        try:
            qn_dict = {int(k.lstrip('q')): v for k, v in qn_kwargs.items()}
        except ValueError:
            raise TypeError("Keyword parameters must have the form 'qN' "
                            "where N is an integer.")
        qn_dict.update(enumerate(qn, start=2))
        self._qn = {k: np.asarray(v, dtype=complex).ravel()
                    for k, v in qn_dict.items()
                    if v is not None}

        self._corr = collections.defaultdict(dict)
        self._corr_err = collections.defaultdict(dict)

    def _get_qn(self, n):
        try:
            return self._qn[n]
        except KeyError:
            raise ValueError(
                'q_{} is required for this calculation but was not provided.'
                .format(n)
            )

    def _calculate_corr(self, n, k, error=False):
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

    def _calculate_corr_err(self, n, k):
        # skip if this error was already calculated
        if k in self._corr_err[n]:
            return

        if k == 2:
            M = self._M
            weights = M*(M-1)
            w1 = np.einsum('i->', weights)  # sum of weights
            w2 = np.inner(weights, weights)  # sum of squared weights

            # event-by-event differences from mean: <2>_i - <<2>>
            qn = self._get_qn(n)
            diff = (np.real(qn*qn.conj())-M)/weights - self._corr[n][k]

            # weighted variance
            var = np.sum(weights*np.square(diff)) / w1

            # unbiased variance s^2
            ssq = var / (1. - w2/(w1*w1))

            # final error of <<2>>
            self._corr_err[n][k] = np.sqrt(w2*ssq)/w1

        else:
            raise ValueError('Error is only implemented for k == 2.')

    def correlation(self, n, k, error=False):
        r"""
        Calculate `\langle k \rangle_n`,
        the `k`-particle correlation function for `n`\ th-order anisotropy.

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error
            (for `\langle 2 \rangle` only).
            If true, return a tuple ``(corr, corr_error)``.

        """
        self._calculate_corr(n, k)
        corr_nk = self._corr[n][k]

        if error:
            self._calculate_corr_err(n, k)
            return corr_nk, self._corr_err[n][k]
        else:
            return corr_nk

    def cumulant(self, n, k, error=False):
        r"""
        Calculate `c_n\{k\}`,
        the `k`-particle cumulant for `n`\ th-order anisotropy.

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error (for `c_n\{2\}` only).
            If true, return a tuple ``(cn2, cn2_error)``.

        """
        corr_nk = self.correlation(n, k, error=error)

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

        :param int n: Anisotropy order.
        :param int k: Correlation order.

        :param bool error:
            Whether to calculate statistical error (for `v_n\{2\}` only).
            If true, return a tuple ``(vn2, vn2_error)``.

        :param str imaginary: (optional)
            Determines behavior when the computed flow is imaginary:

            - ``'nan'`` (default) -- Return NaN and raise a ``RuntimeWarning``.
            - ``'negative'`` -- Return the negative absolute value.
            - ``'zero'`` -- Return ``0.0``.

        """
        cnk = self.cumulant(n, k, error=error)

        if error:
            cnk, cnk_err = cnk

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

        if k == 2 and error:
            return vnk, .5/np.sqrt(abs(cnk)) * cnk_err
        else:
            return vnk


class Sampler(object):
    r"""
    Random flow event generator.

    A ``Sampler`` object represents an event with specified flow coefficients
    `v_n`.  It computes and randomly samples `dN/d\phi` as a probability
    density function (PDF).

    :param float v2, v3, ...:
        Flow coefficients as positional arguments.

    :param float vn_kwargs:
        Flow coefficients as keyword arguments.

    If no ``vn`` arguments are given, ``Sampler`` will have zero flow, i.e.
    ``Sampler.pdf()`` will be flat and ``Sampler.sample()`` will generate
    uniform numbers from `[-\pi, \pi)`.

    """
    def __init__(self, *vn, **vn_kwargs):
        if not vn and not vn_kwargs:
            self._n = self._vn = None
        else:
            try:
                vn_dict = {int(k.lstrip('v')): v for k, v in vn_kwargs.items()}
            except ValueError:
                raise TypeError("Keyword parameters must have the form 'vN' "
                                "where N is an integer.")
            vn_dict.update((k, v) for k, v in enumerate(vn, start=2)
                           if v is not None and v != 0.)
            kwargs = dict(dtype=float, count=len(vn_dict))
            self._n = np.fromiter(vn_dict.keys(), **kwargs)
            self._vn = np.fromiter(vn_dict.values(), **kwargs)

    def _pdf(self, phi):
        """
        Evaluate the _unnormalized_ flow PDF.

        """
        pdf = np.inner(self._vn, np.cos(np.outer(phi, self._n)))
        pdf *= 2.
        pdf += 1.

        return pdf

    @staticmethod
    def _uniform_phi(M):
        """
        Generate M random numbers in [-pi, pi).

        """
        return np.random.uniform(-np.pi, np.pi, M)

    def pdf(self, phi):
        r"""
        Evaluate the flow PDF `dN/d\phi`.

        :param array-like phi: Azimuthal angles.

        :returns: The flow PDF evaluated at ``phi``.

        """
        if self._n is None:
            pdf = np.empty_like(phi)
            pdf.fill(.5/np.pi)
            return pdf

        phi = np.asarray(phi)

        pdf = self._pdf(phi)
        pdf /= 2.*np.pi

        return pdf

    def sample(self, multiplicity):
        r"""
        Randomly sample azimuthal angles `\phi`.

        :param int multiplicity: Number to sample.

        :returns: Array of sampled angles.

        """
        if self._n is None:
            return self._uniform_phi(multiplicity)

        # Since the flow PDF does not have an analytic inverse CDF, I use a
        # simple accept-reject sampling algorithm.  This is reasonably
        # efficient since for normal-sized vn, the PDF is close to flat.  Now
        # due to the overhead of Python functions, it's desirable to minimize
        # the number of calls to the random number generator.  Therefore I
        # sample numbers in chunks; most of the time only one or two chunks
        # should be needed.  Eventually, I might rewrite this with Cython, but
        # it's fast enough for now.

        N = 0  # number of phi that have been sampled
        phi = np.empty(multiplicity)  # allocate array for phi
        pdf_max = 1 + 2*self._vn.sum()  # sampling efficiency ~ 1/pdf_max

        while N < multiplicity:
            n_remaining = multiplicity - N
            n_to_sample = int(1.03*pdf_max*n_remaining)
            phi_chunk = self._uniform_phi(n_to_sample)
            phi_chunk = phi_chunk[self._pdf(phi_chunk) >
                                  np.random.uniform(0, pdf_max, n_to_sample)]
            K = min(phi_chunk.size, n_remaining)  # number of phi to take
            phi[N:N+K] = phi_chunk[:K]
            N += K

        return phi
