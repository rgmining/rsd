#
#  test_graph.py
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

"""Unit tests for rsd.graph module.
"""

import pytest
from numpy import testing

from rsd import ReviewGraph


def test_new_reviewer() -> None:
    """Test for creating reviewers."""
    name1 = "test-reviewer1"
    g = ReviewGraph(0.2)
    r1 = g.new_reviewer(name1, 0.1)
    assert r1.name == name1
    testing.assert_almost_equal(r1.anomalous_score, 0.1)

    name2 = "test-reviewer2"
    r2 = g.new_reviewer(name2)
    assert r2.name == name2
    testing.assert_almost_equal(r2.anomalous_score, 0.5)

    assert len(g.reviewers) == 2
    assert r1 in g.reviewers
    assert r2 in g.reviewers


def test_new_product() -> None:
    """Test for creating products."""
    name = "test-product"
    g = ReviewGraph(0.2)
    p = g.new_product(name)
    assert p.name == name
    assert len(g.products) == 1
    assert p in g.products


def test_add_review() -> None:
    """Test for adding reviews.

    This test uses the following sample graph.

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
    g = ReviewGraph(0.2)
    for i in range(2):
        g.new_reviewer(f"reviewer-{i}")
    for i in range(3):
        g.new_product(f"product-{i}")
    for i, r in enumerate(g.reviewers):
        for p in g.products[i:]:
            rating = 0.1 if i == 0 else 0.8
            review = g.add_review(r, p, rating)
            assert review.rating == rating


@pytest.fixture
def review_graph() -> ReviewGraph:
    """Generate the following sample graph.

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
    g = ReviewGraph(0.2)
    for i in range(2):
        g.new_reviewer(f"reviewer-{i}")
    for i in range(3):
        g.new_product(f"product-{i}")
    for i, r in enumerate(g.reviewers):
        for p in g.products[i:]:
            g.add_review(r, p, 0.1 if i == 0 else 0.8)
    return g


def test_retrieve_reviewers(review_graph: ReviewGraph) -> None:
    """Test retrieve_reviewers method."""
    review = review_graph.reviews[0]  # reviewer-0 -> product-0
    assert review_graph.reviewers[0] in review_graph.retrieve_reviewers(review)


def test_retrieve_products(review_graph: ReviewGraph) -> None:
    """Test retrieve_products method."""
    review = review_graph.reviews[0]  # reviewer-0 -> product-0
    assert review_graph.products[0] in review_graph.retrieve_products(review)


def test_retrieve_reviews_by_reviewer(review_graph: ReviewGraph) -> None:
    """Test retrieve_reviews_by_reviewer method."""
    reviews = review_graph.retrieve_reviews_by_reviewer(review_graph.reviewers[0])
    assert set(reviews) == set(review_graph.reviews[:3])


def test_retrieve_reviews_by_product(review_graph: ReviewGraph) -> None:
    """Test retrieve_reviews_by_product method."""
    reviews = review_graph.retrieve_reviews_by_product(review_graph.products[2])
    assert review_graph.reviews[2] in reviews
    assert review_graph.reviews[4] in reviews


def test_retrieve_reviews(review_graph: ReviewGraph) -> None:
    """test retrieve_reviews method."""
    target = review_graph.reviews[2]
    agree, disagree = review_graph.retrieve_reviews(target)

    assert set(agree) == set()
    assert set(disagree) == {review_graph.reviews[4]}
