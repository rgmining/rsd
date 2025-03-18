#
#  test_reviewer.py
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
"""Test case for Reviewer class.

This test uses the following sample graph.

.. graphviz::

   digraph bipartite {
      graph [rankdir = LR];
      "reviewer-0";
      "product-0";
      "product-1";
      "product-2";
      "reviewer-0" -> "product-0" [label="rating=0.1, honesty=0.1"];
      "reviewer-0" -> "product-1" [label="rating=0.1, honesty=0.2"];
      "reviewer-0" -> "product-2" [label="rating=0.1, honesty=0.5"];
   }

"""

import pytest
from numpy import testing

from rsd import ReviewGraph, graph


@pytest.fixture
def review_graph() -> ReviewGraph:
    g = ReviewGraph(0.2)
    reviewer = g.new_reviewer("reviewer-0", 0.1)
    products = [g.new_product(f"product-{i}") for i in range(3)]

    r = g.add_review(reviewer, products[0], 0.1)
    r.honesty = 0.1

    r = g.add_review(reviewer, products[1], 0.1)
    r.honesty = 0.2

    r = g.add_review(reviewer, products[2], 0.1)
    r.honesty = 0.5

    return g


def test_anomalous_score(review_graph: ReviewGraph) -> None:
    """Test the reviewer has the given anomalous score."""
    testing.assert_almost_equal(review_graph.reviewers[0].anomalous_score, 0.1)


def test_trustiness(review_graph: ReviewGraph) -> None:
    """Test the reviewer has the given trustiness."""
    testing.assert_almost_equal(review_graph.reviewers[0].trustiness, 1 - 0.1)


def test_update_trustiness(review_graph: ReviewGraph) -> None:
    """Test update_trustiness method."""
    new = graph._scale(sum([r.honesty for r in review_graph.reviews]))
    diff = abs(review_graph.reviewers[0].trustiness - new)

    res = review_graph.reviewers[0].update_trustiness()
    testing.assert_almost_equal(res, diff)
    testing.assert_almost_equal(review_graph.reviewers[0].trustiness, new)
