#
#  test_scale.py
#
#  Copyright (c) 2016-2025 Junpei Kawamoto
#
#  This file is part of rgmining-rsd.
#
#  rgmining-rsd is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  rgmining-rsd is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
"""Test case for _scale function.

The scale function is defined as:

.. math::

    {\\rm scale}(v) = \\frac{2}{1 + \\exp(-v)} - 1

"""

from random import random

import numpy as np
from numpy import testing

from rsd import graph


def test() -> None:
    """Test with random values."""
    scale = 10
    for r in (random() * scale - scale / 2 for _ in range(10000)):
        testing.assert_almost_equal(graph._scale(r), 2 / (1 + np.exp(-r)) - 1)


def test_inf() -> None:
    """Test with inf."""
    testing.assert_almost_equal(graph._scale(float("inf")), 1)
    testing.assert_almost_equal(graph._scale(-float("inf")), -1)
