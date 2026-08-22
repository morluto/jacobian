"""Tests for coalgebra operations."""

from __future__ import annotations

import pytest

from jacobian.math.coalgebras._models import (
    GROUP_LIKE_ENUMERATION_BUDGET,
    Coalgebra,
    ComultiplicationRequest,
    CounitRequest,
    GroupLikeElementsRequest,
)
from jacobian.math.coalgebras._operations import (
    compute_comultiplication,
    compute_counit,
    find_group_like_elements,
)


class TestComultiplication:
    """Test comultiplication computation."""

    def test_trivial_group_coalgebra(self):
        """Delta(1) = 1 ⊗ 1 for the trivial group coalgebra."""
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        result = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=0)
        )
        assert result.coefficients[0][0] == 1

    def test_two_dim(self):
        """Compute comultiplication for a 2D coalgebra."""
        ca = Coalgebra(
            prime=7,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        result = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=0)
        )
        assert result.coefficients[0][0] == 1
        assert result.coefficients[1][1] == 0


class TestCounit:
    """Test counit computation."""

    def test_counit_unity(self):
        """epsilon(1) = 1 for the group-like element."""
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        result = compute_counit(CounitRequest(coalgebra=ca, element_index=0))
        assert result.value == 1

    def test_counit_second_group_like(self):
        """epsilon(e2) = 1 in the two-group-like coalgebra."""
        ca = Coalgebra(
            prime=5,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        result = compute_counit(CounitRequest(coalgebra=ca, element_index=1))
        assert result.value == 1


class TestGroupLikeElements:
    """Test group-like element finding."""

    def test_trivial_group(self):
        """The trivial group coalgebra has 1 group-like element."""
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 1

    def test_scaled_group_like_found(self):
        """Delta(c)=2c tensor c with epsilon(c)=3 admits the scaled group-like 2c."""
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((2,),),),
            counit=(3,),
        )
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 1

    def test_two_group_like(self):
        """A coalgebra with two group-like elements."""
        ca = Coalgebra(
            prime=5,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 2

    def test_scaled_group_like(self):
        """A scaled basis vector can be group-like: g = 2c over GF(5).

        With Delta(c) = 2 c (x) c and epsilon(c) = 3, g = 2c satisfies
        epsilon(g) = 2*3 = 1 (mod 5) and Delta(g) = 4 c (x) c = g (x) g,
        while no basis element is group-like.
        """
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((2,),),),
            counit=(3,),
        )
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 1
        assert result.elements[0].coefficients == (2,)

    def test_composite_prime_rejected(self):
        """A composite modulus is not a field and must be rejected."""
        with pytest.raises(ValueError, match="prime must be a prime integer"):
            Coalgebra(
                prime=4,
                dimension=1,
                comultiplication=(((1,),),),
                counit=(1,),
            )

    def test_enumeration_budget_rejected(self):
        """Requests whose element space exceeds the budget are rejected."""
        ca = Coalgebra(
            prime=263,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert ca.prime ** ca.dimension > GROUP_LIKE_ENUMERATION_BUDGET
        with pytest.raises(ValueError, match="enumeration requires"):
            GroupLikeElementsRequest(coalgebra=ca)

    def test_within_enumeration_budget_admitted(self):
        """An element space inside the budget enumerates exhaustively."""
        ca = Coalgebra(
            prime=251,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert ca.prime ** ca.dimension <= GROUP_LIKE_ENUMERATION_BUDGET
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        # Both basis elements are group-like in this direct-sum coalgebra.
        assert result.count == 2
        found = {tuple(e.coefficients) for e in result.elements}
        assert found == {(1, 0), (0, 1)}
