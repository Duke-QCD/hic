# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

__all__ = 'IC',


class IC(object):
    """
    Initial condition profile.

    profile: (nx, ny)
        The IC profile as a block-style grid.  It is assumed that the
        middle of the array corresponds to (x, y) == (0, 0).
    dxy: float or pair of floats
        Grid spacing, either a single value dxy = dx = dy or a pair
        dxy = (dx, dy).

    """
    def __init__(self, profile, dxy):
        self._profile = np.asarray(profile, dtype=float)

        # save (x, y) steps
        try:
            self._dx, self._dy = dxy
        except (TypeError, ValueError):
            self._dx = self._dy = dxy

        # save (x, y) max
        ny, nx = self._profile.shape
        xmax = .5*self._dx*(nx - 1.)
        ymax = .5*self._dy*(ny - 1.)
        self._xymax = xmax, ymax

        # calculate and save center of mass
        X = np.linspace(-xmax, xmax, nx)
        Y = np.linspace(ymax, -ymax, ny)
        cm = np.array((
            np.inner(X, self._profile.sum(axis=0)),
            np.inner(Y, self._profile.sum(axis=1))
        ))
        cm /= self._profile.sum()
        self._cm = cm

    def sum(self):
        """
        Sum of profile scaled by grid cell size.

        """
        return self._profile.sum() * self._dx * self._dy

    def cm(self):
        """
        Center of mass coordinates.

        """
        return self._cm

    def ecc(self, n):
        """
        Eccentricity epsilon_n.

        """
        ny, nx = self._profile.shape
        xmax, ymax = self._xymax
        xcm, ycm = self._cm

        # create (X, Y) grids relative to CM
        Y, X = np.mgrid[ymax:-ymax:1j*ny, -xmax:xmax:1j*nx]
        X -= xcm
        Y -= ycm

        # create grid of weights = profile * R^n
        Rsq = X*X + Y*Y
        if n == 1:
            W = np.sqrt(Rsq, out=Rsq)
        elif n == 2:
            W = Rsq
        else:
            if n & 1:  # odd n
                W = np.sqrt(Rsq)
            else:  # even n
                W = Rsq
            # multiply by R^2 until W = R^n
            for _ in range(int((n-1)/2)):
                W *= Rsq
        W *= self._profile

        # create grid of e^{i*n*phi} * W
        i_n_phi = np.zeros_like(X, dtype=complex)
        np.arctan2(Y, X, out=i_n_phi.imag)
        i_n_phi.imag *= n
        exp_phi = np.exp(i_n_phi, out=i_n_phi)
        exp_phi *= W

        return abs(exp_phi.sum()) / W.sum()
