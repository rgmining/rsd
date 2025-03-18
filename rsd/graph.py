#
#  graph.py
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
"""Implementation of RSD."""

from collections.abc import Collection
from functools import cache
from typing import Final, NamedTuple, Optional

import networkx as nx
import numpy as np


def _scale(v: float) -> float:
    """Scaling a given value.

    The output is defined by

    .. math::

        {\\rm scale}(v) = \\frac{2}{1 + \\exp(-v)} - 1

    Args:
      v: Input value.

    Returns:
      Output value defined above.
    """
    e = np.exp(-v)
    return float(2.0 / (1.0 + e) - 1.0)


class Node:
    """Abstract class of review graph.

    Args:
      graph: the graph object this node will belong to.
      name: name of this node.
    """

    name: Final[str]
    """Name of this node."""
    _g: Final["ReviewGraph"]
    """Graph object this node belongs to."""

    __slots__ = ("name", "_g")

    def __init__(self, graph: "ReviewGraph", name: Optional[str] = None) -> None:
        self._g = graph
        self.name = name if name else super().__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return 13 * hash(type(self)) + 17 * hash(self.name)


class Reviewer(Node):
    """A node class representing a reviewer.

    Args:
      graph: Graph object this reviewer belongs to.
      name: Name of this reviewer.
      anomalous: Initial anomalous score (default: None).
    """

    trustiness: float
    """A float value in [0, 1] which represents trustiness of this reviewer."""

    __slots__ = ("trustiness",)

    def __init__(self, graph: "ReviewGraph", name: Optional[str] = None, anomalous: Optional[float] = None) -> None:
        super().__init__(graph, name)

        # If an initial anomalous score is given, use it.
        self.trustiness = 1.0 - anomalous if anomalous else 0.5

    @property
    def anomalous_score(self) -> float:
        """Returns the anomalous score of this reviewer.

        The anomalous score is defined by 1 - trustiness.
        """
        return 1.0 - self.trustiness

    def update_trustiness(self) -> float:
        """Update trustiness of this reviewer.

        The updated trustiness of a reviewer :math:`u` is defined by

        .. math::

           {\\rm trustiness}(u) =
             \\frac{2}{1 + \\exp(-\\sum_{r \\in R(u)} {\\rm honesty(r)} )} - 1

        where :math:`R(u)` is a set of reviews the reviewer :math:`u` posts.

        Returns;
          absolute difference between the old trustiness and updated one.
        """
        sum_h = 0.0
        for re in self._g.retrieve_reviews_by_reviewer(self):
            sum_h += re.honesty
        new = _scale(sum_h)

        diff = abs(self.trustiness - new)
        self.trustiness = new
        return diff

    def __str__(self) -> str:
        return f"{self.name}: {self.anomalous_score}"


class Product(Node):
    """A node class representing a product.

    Args:
      graph: Graph object this product belongs to.
      name: Name of this product.
    """

    reliability: float
    """A float value in [0, 1], which represents reliability of this product."""

    __slots__ = ("reliability",)

    def __init__(self, graph: "ReviewGraph", name: Optional[str] = None):
        super(Product, self).__init__(graph, name)
        self.reliability = 0.5

    @property
    def summary(self) -> float:
        """Summary of reviews.

        This value is same as reliability.
        Original algorithm uses *reliability* but our algorithm uses *summary*.
        For convenience, both properties remain.
        """
        return self.reliability

    def update_reliability(self) -> float:
        """Update product's reliability.

        The new reliability is defined by

        .. math::

          {\\rm reliability}(p) = \\frac{2}{1 + e^{-\\theta}} - 1, \\quad
          \\theta = \\sum_{r \\in R(p)}
                      {\\rm trustiness}(r)({\\rm review}(r, p) - \\hat{s}),

        where :math:`R(p)` is a set of reviewers product *p* receives,
        trustiness is defined in :meth:`Reviewer.trustiness`,
        review(*r*, *p*) is the review score reviewer *r* has given to product *p*,
        and :math:`\\hat{s}` is the median of review scores.

        Returns:
          absolute difference between old reliability and new one.
        """
        res = 0.0

        reviews = self._g.retrieve_reviews_by_product(self)
        s = float(np.median([re.rating for re in reviews]))
        for re in reviews:
            for r in self._g.retrieve_reviewers(re):
                res += r.trustiness * (re.rating - s)

        new = _scale(res)
        diff = abs(self.reliability - new)
        self.reliability = new
        return diff

    def __str__(self) -> str:
        return f"{self.name}: {self.summary}"


class Review:
    """A graph entity representing a review.

    Args:
      graph: Graph object this product belongs to.
      time: When this review is posted.
      rating: Rating of this review.
    """

    rating: Final[float]
    """Rating score of this review."""
    honesty: float
    """Honesty score."""
    agreement: float
    """Agreement score."""
    time: Final[int]
    """Time when this review posted."""

    _g: Final["ReviewGraph"]

    __slots__ = ("rating", "honesty", "agreement", "time", "_g")

    def __init__(self, graph: "ReviewGraph", time: int, rating: float) -> None:
        self._g = graph
        self.time = time
        self.rating = rating

        self.honesty = 0.5
        self.agreement = 0.5

    def update_honesty(self) -> float:
        """Update honesty of this review.

        The updated honesty of this review :math:`r` is defined by

        .. math::

           {\\rm honesty}(r)
             = |{\\rm reliability}(P(r))| \\times {\\rm agreement}(r)

        where :math:`P(r)` is the product this review posted.

        Returns:
          absolute difference between old honesty and new one.
        """
        res = 0.0
        for p in self._g.retrieve_products(self):
            res += abs(p.reliability) * self.agreement

        diff = abs(self.honesty - res)
        self.honesty = res
        return diff

    def update_agreement(self, delta: float) -> float:
        """Update agreement of this review.

        This process considers reviews posted in a close time span of this review.
        More precisely, let :math:`t` be the time when this review posted
        and :math:`\\delta` be the time span,
        only reviews of which posted times are in :math:`[t - \\delta, t+\\delta]`
        are considered.

        The updated agreement of a review :math:`r` will be computed with such
        reviews by

        .. math::

           {\\rm agreement}(r)
             = \\frac{2}{1 + \\exp(
                \\sum_{v \\in R_{+}} {\\rm trustiness}(v)
                    - \\sum_{v \\in R_{-}} {\\rm trustiness}(v)
             )} - 1

        where :math:`R_{+}` is a set of reviews close to the review :math:`r`,
        i.e. the difference between ratings are smaller than or equal to delta,
        :math:`R_{-}` is the other reviews. The trustiness of a review means
        the trustiness of the reviewer who posts the review.

        Args:
          delta: a time span :math:`\\delta`.
                 Only reviews posted in the span will be considered for this update.

        Returns:
          absolute difference between old agreement and new one.
        """
        score_diff = 1.0 / 5.0
        agree, disagree = self._g.retrieve_reviews(self, delta, score_diff)

        res = 0.0
        for re in agree:
            for r in self._g.retrieve_reviewers(re):
                res += r.trustiness
        for re in disagree:
            for r in self._g.retrieve_reviewers(re):
                res -= r.trustiness

        new = _scale(res)
        diff = abs(self.agreement - new)
        self.agreement = new
        return diff

    def __str__(self) -> str:
        return f"Review (time={self.time}, rating={self.rating}, agreement={self.agreement}, honesty={self.honesty})"


class ReviewSet(NamedTuple):
    """Pair of agreed reviews and disagreed reviews."""

    agree: Collection[Review]
    """Collection of agreed reviews."""
    disagree: Collection[Review]
    """Collection of disagreed reviews."""


class ReviewGraph:
    """A bipartite graph of which one set of nodes represent reviewers and the other set of nodes represent products.

    Each edge has a label representing a review.

    Args:
        theta: A parameter for updating.
            See `the paper <https://ieeexplore.ieee.org/document/6137345?arnumber=6137345>`__ for more details.
    """

    graph: Final[nx.DiGraph]
    """Graph object of networkx."""
    reviewers: Final[list[Reviewer]]
    """Collection of reviewers."""
    products: Final[list[Product]]
    """Collection of products."""
    reviews: Final[list[Review]]
    """Collection of reviews."""
    _theta: Final[float]
    """Parameter of the algorithm."""
    _delta: Optional[float]
    """Cached time delta."""

    def __init__(self, theta: float) -> None:
        self.graph = nx.DiGraph()
        self.reviewers = []
        self.products = []
        self.reviews = []
        self._theta = theta
        self._delta = None

    @property
    def delta(self) -> float:
        """Time delta.

        This value is defined by
        :math:`\\delta = (t_{\\rm max} - t_{\\rm min}) \\times \\theta`,
        where :math:`t_{\\rm max}, t_{\\rm min}` are the maximum time,
        minimum time of all reviews, respectively,
        :math:`\\theta` is the given parameter defining time ratio.
        """
        if not self._delta:
            min_time = min([r.time for r in self.reviews])
            max_time = max([r.time for r in self.reviews])
            self._delta = (max_time - min_time) * self._theta
        return self._delta

    def new_reviewer(self, name: Optional[str] = None, anomalous: Optional[float] = None) -> Reviewer:
        """Create a new reviewer.

        Args:
            name: the name of the new reviewer.
            anomalous: the anomalous score of the new reviewer.

        Returns:
            A new reviewer instance.
        """
        n = Reviewer(self, name=name, anomalous=anomalous)
        self.graph.add_node(n)
        self.reviewers.append(n)
        return n

    def new_product(self, name: Optional[str] = None) -> Product:
        """Create a new product.

        Args:
            name: The name of the new product.

        Returns:
            A new product instance.
        """
        n = Product(self, name)
        self.graph.add_node(n)
        self.products.append(n)
        return n

    def add_review(self, reviewer: Reviewer, product: Product, review: float, time: Optional[int] = None) -> Review:
        """Add a new review.

        Args:
          reviewer:  An instance of Reviewer.
          product: An instance of Product.
          review: A real number representing review score.
          time: An integer representing reviewing time. (optional)

        Returns:
          the new review object.
        """
        if not time:
            re = Review(self, len(self.reviews), review)
        else:
            re = Review(self, time, review)
        self.graph.add_node(re)
        self.reviews.append(re)
        self.graph.add_edge(reviewer, re)
        self.graph.add_edge(re, product)
        self._delta = None
        return re

    @cache
    def retrieve_reviewers(self, review: Review) -> Collection[Reviewer]:
        """Find reviewers associated with a review.

        Args:
            review: A review instance.

        Returns:
            A list of reviewers associated with the review.
        """
        return list(self.graph.predecessors(review))

    @cache
    def retrieve_products(self, review: Review) -> Collection[Product]:
        """Find products associated with a review.

        Args:
            review: A review instance.

        Returns:
            A list of products associated with the given review.
        """
        return list(self.graph.successors(review))

    @cache
    def retrieve_reviews_by_reviewer(self, reviewer: Reviewer) -> Collection[Review]:
        """Find reviews given by a reviewer.

        Args:
            reviewer: Reviewer

        Returns:
            A list of reviews given by the reviewer.
        """
        return list(self.graph.successors(reviewer))

    @cache
    def retrieve_reviews_by_product(self, product: Product) -> Collection[Review]:
        """Find reviews to a product.

        Args:
            product: Product

        Returns:
            A list of reviews to the product.
        """
        return list(self.graph.predecessors(product))

    def retrieve_reviews(
        self, review: Review, time_diff: Optional[float] = None, score_diff: float = 0.25
    ) -> ReviewSet:
        """Find agree and disagree reviews.

        This method retrieve two groups of reviews.
        Agree reviews have similar scores to a given review.
        On the other hands disagree reviews have different scores.

        Args:
          review: A review instance.
          time_diff: An integer.
          score_diff: An float value.

        Returns:
          A tuple consists of (a list of agree reviews, a list of disagree reviews)
        """
        if not time_diff:
            time_diff = float("inf")

        agree, disagree = [], []
        for p in self.retrieve_products(review):
            for re in self.retrieve_reviews_by_product(p):
                if re == review:
                    continue
                if abs(re.time - review.time) < time_diff:
                    if abs(re.rating - review.rating) < score_diff:
                        agree.append(re)
                    else:
                        disagree.append(re)
        return ReviewSet(agree, disagree)

    def update(self) -> float:
        """Update reviewers' anomalous scores and products' summaries.

        This update process consists of four steps;

        1. Update honesty of reviews (See also :meth:`Review.update_honesty`),
        2. Update rustiness of reviewers
           (See also :meth:`Reviewer.update_trustiness`),
        3. Update reliability of products
           (See also :meth:`Product.update_reliability`),
        4. Update agreements of reviews
           (See also :meth:`Review.update_agreement`).

        Returns:
          summation of maximum absolute updates for the above four steps.
        """
        diff = max(re.update_honesty() for re in self.reviews)
        diff += max(r.update_trustiness() for r in self.reviewers)
        diff += max(p.update_reliability() for p in self.products)
        diff += max(re.update_agreement(self.delta) for re in self.reviews)
        return diff
