#
#  test_review.py
#
#  Copyright (c) 2016-2022 Junpei Kawamoto
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
"""Test case for Review class.

This test case uses the following sample graph.

.. graphviz::

   digraph bipartite {
      graph [rankdir = LR];
      "reviewer-0";
      "reviewer-1";
      "product-0";
      "product-1";
      "product-2";
      "reviewer-0" -> "product-0" [label="0.1"];
      "reviewer-0" -> "product-1" [label="0.1"];
      "reviewer-0" -> "product-2" [label="0.1"];
      "reviewer-1" -> "product-1" [label="0.8"];
      "reviewer-1" -> "product-2" [label="0.8"];
   }

"""
import pytest
from numpy import testing

from rsd import ReviewGraph, graph


@pytest.fixture
def review_graph() -> ReviewGraph:
    g = ReviewGraph(0.2)
    for i in range(2):
        g.new_reviewer(f"reviewer-{i}")
    for i in range(3):
        g.new_product(f"product-{i}")
    for i, r in enumerate(g.reviewers):
        for p in g.products[i:]:
            g.add_review(r, p, 0.1 if i == 0 else 0.8)
    return g


def test_update_honesty(review_graph: ReviewGraph) -> None:
    """Test update_honesty method."""
    target = review_graph.reviews[0]
    expected = abs(review_graph.products[0].reliability) * target.agreement
    diff = abs(target.honesty - expected)
    testing.assert_almost_equal(target.update_honesty(), diff)
    testing.assert_almost_equal(target.honesty, expected)


def test_update_agreement(review_graph: ReviewGraph) -> None:
    """Test update_agreement method."""
    target = review_graph.reviews[2]
    expected = graph._scale(-review_graph.reviewers[1].trustiness)
    diff = abs(target.agreement - expected)
    testing.assert_almost_equal(target.update_agreement(1000), diff)
    testing.assert_almost_equal(target.agreement, expected)
