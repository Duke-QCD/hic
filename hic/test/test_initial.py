# -*- coding: utf-8 -*-

from __future__ import division

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
