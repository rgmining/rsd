#
#  __init__.py
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
"""Review Graph Based Online Store Review Spammer Detection.

RSD is an algorithm introduced by Guan Wang, et al. in ICDM2011_.
This algorithm represents review data as a following graph.

.. graphviz::

  digraph bipartite {
    graph [label="Graph model used in RSD.", rankdir = LR];
    "r1" [label="Reviewer 1 \n (trustiness: 0.1)"];
    "r2" [label="Reviewer 2 \n (trustiness: 0.9)"];
    "r3" [label="Reviewer 3 \n (trustiness: 0.5)"];
    "p1" [label="Product 1 \n (reliability: 0.3)"];
    "p2" [label="Product 2 \n (reliability: 0.8)"];
    "r1p1" [label="0.3"];
    "r1p2" [label="0.9"];
    "r2p2" [label="0.1"];
    "r3p2" [label="0.5"];
    "r1" -> "r1p1" -> "p1";
    "r1" -> "r1p2" -> "p2";
    "r2" -> "r2p2" -> "p2";
    "r3" -> "r3p2" -> "p2";
    "d_r1p1" [shape=box, label="time: 1 \n honesty: 0.4 \n agreement: 1.0 "];
    "d_r1p2" [shape=box, label="time: 4 \n honesty: 0.1 \n agreement: 0.3 "];
    "d_r2p2" [shape=box, label="time: 2 \n honesty: 0.8 \n agreement: 0.3 "];
    "d_r3p2" [shape=box, label="time: 3 \n honesty: 0.2 \n agreement: 0.3 "];
    "r1p1" -> "d_r1p1" [style=dotted];
    "r1p2" -> "d_r1p2" [style=dotted];
    "r2p2" -> "d_r2p2" [style=dotted];
    "r3p2" -> "d_r3p2" [style=dotted];
  }

This package exports `ReviewGraph` which is an alias of :class:`rsd.graph.ReviewGraph`.

.. _ICDM2011: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6137345

"""

from typing import Final

from rsd.graph import ReviewGraph

__version__: Final = "0.3.4"

__all__: Final = ("ReviewGraph", "__version__")
