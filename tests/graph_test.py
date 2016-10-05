#
# graph_test.py
#
# Copyright (c) 2016 Junpei Kawamoto
#
# This file is part of rgmining-rsd.
#
# rgmining-rsd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# rgmining-rsd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
"""Unit tests for rsd.graph module.
"""
# pylint: disable=protected-access
import numpy as np
import random
import unittest
from rsd import graph


class TestScale(unittest.TestCase):
    """Test case for _scale function.

    The scale function is defeined as:

    .. math::

        {\\rm scale}(v) = \\frac{2}{1 + \\exp(-v)} - 1

    """

    def test(self):
        """Test with random values.
        """
        SCALE = 10
        for r in [random.random() * SCALE - SCALE / 2 for _ in range(10000)]:
            self.assertAlmostEqual(graph._scale(r), 2 / (1 + np.exp(-r)) - 1)

    def test_inf(self):
        """Test with inf.
        """
        self.assertAlmostEqual(graph._scale(float("inf")), 1)
        self.assertAlmostEqual(graph._scale(-float("inf")), -1)


class TestReviewer(unittest.TestCase):
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

    def setUp(self):
        """Set up for tests.
        """
        self.graph = graph.ReviewGraph(0.2)
        self.reviewer = self.graph.new_reviewer("reviewer-0", 0.1)
        self.products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)]

        self.reviews = []

        r = self.graph.add_review(self.reviewer, self.products[0], 0.1)
        r.honesty = 0.1
        self.reviews.append(r)

        r = self.graph.add_review(self.reviewer, self.products[1], 0.1)
        r.honesty = 0.2
        self.reviews.append(r)

        r = self.graph.add_review(self.reviewer, self.products[2], 0.1)
        r.honesty = 0.5
        self.reviews.append(r)

    def test_anomalous_score(self):
        """Test the reviewer has the given anomalous score.
        """
        self.assertAlmostEqual(self.reviewer.anomalous_score, 0.1)

    def test_trustiness(self):
        """Test the reviewer has the given trustiness.
        """
        self.assertAlmostEqual(self.reviewer.trustiness, 1 - 0.1)

    def test_update_trustiness(self):
        """Test update_trustiness method.
        """
        new = graph._scale(sum([r.honesty for r in self.reviews]))
        diff = abs(self.reviewer.trustiness - new)

        res = self.reviewer.update_trustiness()
        self.assertAlmostEqual(res, diff)
        self.assertAlmostEqual(self.reviewer.trustiness, new)


class TestProduct(unittest.TestCase):
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

    def setUp(self):
        """Construct the above graph.
        """
        self.graph = graph.ReviewGraph(0.2)

        self.reviewers = []
        self.reviewers.append(self.graph.new_reviewer("reviewer-0", 1 - 0.1))
        self.reviewers.append(self.graph.new_reviewer("reviewer-1", 1 - 0.5))
        self.reviewers.append(self.graph.new_reviewer("reviewer-2", 1 - 0.8))

        self.product = self.graph.new_product("product-0")

        self.reviews = []
        self.reviews.append(
            self.graph.add_review(self.reviewers[0], self.product, 0.3))
        self.reviews.append(
            self.graph.add_review(self.reviewers[1], self.product, 0.2))
        self.reviews.append(
            self.graph.add_review(self.reviewers[2], self.product, 0.8))

    def test_reliability(self):
        """Test reliability.
        """
        self.assertAlmostEqual(self.product.reliability, 0.5)

    def test_summary(self):
        """Test summary.
        """
        self.assertAlmostEqual(self.product.summary, 0.5)

        self.product.reliability = 0.2
        self.assertAlmostEqual(self.product.summary, 0.2)

    def test_update_reliability(self):
        """Test update_reliability.
        """
        s = np.median([review.rating for review in self.reviews])
        theta = 0.
        for i, r in enumerate(self.reviewers):
            theta += r.trustiness * (self.reviews[i].rating - s)

        new = graph._scale(theta)
        diff = abs(self.product.reliability - new)

        self.assertAlmostEqual(self.product.update_reliability(), diff)
        self.assertAlmostEqual(self.product.reliability, new)


class TestReviewGraphCreation(unittest.TestCase):
    """Test case for creating a ReviewGraph.
    """

    def setUp(self):
        """Set up for tests.
        """
        self.graph = graph.ReviewGraph(0.2)

    def test_new_reviewer(self):
        """Test for creating reviewers.
        """
        name1 = "test-reviewer1"
        r1 = self.graph.new_reviewer(name1, 0.1)
        self.assertEqual(r1.name, name1)
        self.assertAlmostEqual(r1.anomalous_score, 0.1)

        name2 = "test-reviewer2"
        r2 = self.graph.new_reviewer(name2)
        self.assertEqual(r2.name, name2)
        self.assertAlmostEqual(r2.anomalous_score, 0.5)

        self.assertEqual(len(self.graph.reviewers), 2)
        self.assertIn(r1, self.graph.reviewers)
        self.assertIn(r2, self.graph.reviewers)

    def test_new_product(self):
        """Test for creating products.
        """
        name = "test-product"
        p = self.graph.new_product(name)
        self.assertEqual(p.name, name)
        self.assertEqual(len(self.graph.products), 1)
        self.assertIn(p, self.graph.products)

    def test_add_review(self):
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
        reviewers = [
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(2)]
        products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)]
        for i, r in enumerate(reviewers):
            for j in range(i, len(products)):
                rating = 0.1 if i == 0 else 0.8
                review = self.graph.add_review(r, products[j], rating)
                self.assertEqual(review.rating, rating)


class TestReview(unittest.TestCase):
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

    def setUp(self):
        """Set up for tests.
        """
        self.graph = graph.ReviewGraph(0.2)
        self.reviewers = [
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(2)]
        self.products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)]
        self.reviews = []
        for i, r in enumerate(self.reviewers):
            for j in range(i, len(self.products)):
                self.reviews.append(self.graph.add_review(
                    r, self.products[j], 0.1 if i == 0 else 0.8))

    def test_update_honesty(self):
        """Test update_honesty method.
        """
        target = self.reviews[0]
        expected = abs(self.products[0].reliability) * target.agreement
        diff = abs(target.honesty - expected)
        self.assertAlmostEqual(target.update_honesty(), diff)
        self.assertAlmostEqual(target.honesty, expected)

    def test_update_agreement(self):
        """Test update_agreement method.
        """
        target = self.reviews[2]
        expected = graph._scale(-self.reviewers[1].trustiness)
        diff = abs(target.agreement - expected)
        self.assertAlmostEqual(target.update_agreement(1000), diff)
        self.assertAlmostEqual(target.agreement, expected)


class TestReviewGraphRetrieval(unittest.TestCase):
    """Test case for retrieving elements in ReviewGraph.

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

    def setUp(self):
        """Set up for tests.
        """
        self.graph = graph.ReviewGraph(0.2)
        self.reviewers = [
            self.graph.new_reviewer("reviewer-{0}".format(i)) for i in range(2)]
        self.products = [
            self.graph.new_product("product-{0}".format(i)) for i in range(3)]
        self.reviews = []
        for i, r in enumerate(self.reviewers):
            for j in range(i, len(self.products)):
                self.reviews.append(self.graph.add_review(
                    r, self.products[j], 0.1 if i == 0 else 0.8))

    def test_retrieve_reviewers(self):
        """Test retrieve_reviewers method.
        """
        review = self.reviews[0]  # reviewer-0 -> product-0
        self.assertIn(self.reviewers[0], self.graph.retrieve_reviewers(review))

        with self.assertRaises(ValueError):
            self.graph.retrieve_reviewers(None)

    def test_retrieve_products(self):
        """Test retrieve_products method.
        """
        review = self.reviews[0]  # reviewer-0 -> product-0
        self.assertIn(self.products[0], self.graph.retrieve_products(review))

        with self.assertRaises(ValueError):
            self.graph.retrieve_products(None)

    def test_retrieve_reviews_by_reviewer(self):
        """Test retrieve_reviews_by_reviewer method.
        """
        reviews = self.graph.retrieve_reviews_by_reviewer(self.reviewers[0])
        self.assertEqual(set(reviews), set(self.reviews[:3]))

        with self.assertRaises(ValueError):
            self.graph.retrieve_reviews_by_reviewer(None)

    def test_retrieve_reviews_by_product(self):
        """Test retrieve_reviews_by_product method.
        """
        reviews = self.graph.retrieve_reviews_by_product(self.products[2])
        self.assertIn(self.reviews[2], reviews)
        self.assertIn(self.reviews[4], reviews)

        with self.assertRaises(ValueError):
            self.graph.retrieve_reviews_by_product(None)

    def test_retrieve_reviews(self):
        """test retrieve_reviews method.
        """
        target = self.reviews[2]
        agree, disagree = self.graph.retrieve_reviews(target)

        self.assertEqual(set(agree), set())
        self.assertEqual(set(disagree), set([self.reviews[4]]))


if __name__ == "__main__":
    unittest.main()
