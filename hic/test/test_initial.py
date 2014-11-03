# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .. import initial


def test_ic():
    """initial conditions"""

    ic = initial.IC((
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 0, 0, 0),
    ), 1)

    assert_almost_equal(ic.sum(), 4, err_msg='Incorrect total profile.')

    assert_array_almost_equal(ic.cm(), (0, 0),
                              err_msg='Incorrect center of mass.')

    assert_almost_equal(ic.ecc(2), 0, err_msg='Incorrect epsilon_2.')
    assert_almost_equal(ic.ecc(3), 0, err_msg='Incorrect epsilon_3.')
    assert_almost_equal(ic.ecc(4), 1, err_msg='Incorrect epsilon_4.')

    ic = initial.IC((
        (0, 0, 0, 0, 0),
        (0, 2, 0, 1, 0),
        (0, 0, 0, 0, 0),
        (0, 3, 0, 4, 0),
        (0, 0, 0, 0, 0),
    ), .1)

    assert_almost_equal(ic.sum(), .1, err_msg='Incorrect total profile.')

    assert_array_almost_equal(ic.cm(), (0, -.04),
                              err_msg='Incorrect center of mass.')

    S = np.array((1, 2, 3, 4), dtype=float)
    X = np.array((.1, -.1, -.1, .1))
    Y = np.array((.1, .1, -.1, -.1))
    Y += .04
    Rsq = X*X + Y*Y

    for n in 2, 3, 4:
        ecc = abs(np.average(
            np.exp(1j*n*np.arctan2(Y, X)),
            weights=S*Rsq**(.5*n)
        ))
        assert_almost_equal(ic.ecc(n), ecc,
                            err_msg='Incorrect epsilon_{}.'.format(n))

    # TODO: test non-square grids and grid spacing
