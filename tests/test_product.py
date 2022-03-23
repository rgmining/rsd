#
#  test_product.py
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
"""Test case for Product class.

This test case uses the following sample graph.

.. graphviz::

   digraph bipartite {
      graph [rankdir = LR];
      "reviewer-0" [label="reviewer-1 \n trustiness=0.1"];
      "reviewer-1" [label="reviewer-1 \n trustiness=0.5"];
      "reviewer-2" [label="reviewer-1 \n trustiness=0.8"];
      "product-0";
      "reviewer-0" -> "product-0" [label="rating=0.3"];
      "reviewer-1" -> "product-0" [label="rating=0.2"];
      "reviewer-2" -> "product-0" [label="rating=0.8"];
   }

"""
import numpy as np
import pytest
from numpy import testing

from rsd import ReviewGraph, graph


@pytest.fixture
def review_graph() -> ReviewGraph:
    """Construct the above graph."""
    g = ReviewGraph(0.2)
    g.new_reviewer("reviewer-0", 1 - 0.1)
    g.new_reviewer("reviewer-1", 1 - 0.5)
    g.new_reviewer("reviewer-2", 1 - 0.8)

    product = g.new_product("product-0")
    g.add_review(g.reviewers[0], product, 0.3)
    g.add_review(g.reviewers[1], product, 0.2)
    g.add_review(g.reviewers[2], product, 0.8)

    return g


def test_reliability(review_graph: ReviewGraph) -> None:
    """Test reliability."""
    testing.assert_almost_equal(review_graph.products[0].reliability, 0.5)


def test_summary(review_graph: ReviewGraph) -> None:
    """Test summary."""
    product = review_graph.products[0]
    testing.assert_almost_equal(product.summary, 0.5)

    product.reliability = 0.2
    testing.assert_almost_equal(product.summary, 0.2)


def test_update_reliability(review_graph: ReviewGraph) -> None:
    """Test update_reliability."""
    s = float(np.median([review.rating for review in review_graph.reviews]))
    theta = 0.0
    for i, r in enumerate(review_graph.reviewers):
        theta += r.trustiness * (review_graph.reviews[i].rating - s)

    product = review_graph.products[0]
    new = graph._scale(theta)
    diff = abs(product.reliability - new)

    testing.assert_almost_equal(product.update_reliability(), diff)
    testing.assert_almost_equal(product.reliability, new)
