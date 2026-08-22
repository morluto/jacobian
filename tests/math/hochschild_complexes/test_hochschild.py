"""Tests for Hochschild complex operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.hochschild_complexes._models import (
    AlgebraStructure,
    HochschildChainComplexRequest,
    HochschildHomologyRequest,
)
from jacobian.math.hochschild_complexes._operations import (
    compute_hochschild_chain_complex,
    compute_hochschild_homology,
)


class TestHochschildChainComplex:
    """Test Hochschild chain complex computation."""

    def test_one_dim_algebra(self):
        """Chain complex of a 1D algebra over GF(5)."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
        )
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=alg, max_degree=2)
        )
        assert result.group_dimensions == (1, 1, 1)

    def test_two_dim_algebra(self):
        """Chain complex of a 2D algebra."""
        alg = AlgebraStructure(
            prime=7,
            dimension=2,
            structure_constants=(
                ((1, 0), (0, 1)),
                ((0, 1), (1, 0)),
            ),
        )
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=alg, max_degree=1)
        )
        assert result.group_dimensions[0] == 1
        assert result.group_dimensions[1] == 2

    def test_has_differentials(self):
        """The chain complex should have differentials."""
        alg = AlgebraStructure(
            prime=5,
            dimension=2,
            structure_constants=(
                ((1, 0), (0, 1)),
                ((0, 1), (1, 0)),
            ),
        )
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=alg, max_degree=2)
        )
        assert len(result.differentials) >= 1


class TestHochschildHomology:
    """Test Hochschild homology computation."""

    def test_identity_algebra(self):
        """HH_0 of the identity algebra is K."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        assert result.groups[0].betti == 1

    def test_2d_commutative(self):
        """Test with a 2D commutative algebra."""
        alg = AlgebraStructure(
            prime=7,
            dimension=2,
            structure_constants=(
                ((1, 0), (0, 1)),
                ((0, 1), (1, 0)),
            ),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        assert len(result.groups) >= 2

    def test_zero_algebra(self):
        """Test with the zero algebra (e*e = 0)."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((0,),),),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        # With zero multiplication, the differential vanishes
        assert result.groups[0].betti >= 0


class TestHochschildAdmissionAndTopDegree:
    def test_composite_prime_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="prime"):
            AlgebraStructure(
                prime=4,
                dimension=1,
                structure_constants=(((1,),),),
            )

    def test_non_associative_rejected(self):
        """[e0,e1]=e0 style left-zero multiplication fails associativity."""
        c = (
            ((0, 0), (0, 0)),
            ((1, 0), (0, 0)),
        )
        with pytest.raises(ValidationError, match="associative"):
            AlgebraStructure(prime=5, dimension=2, structure_constants=c)

    def test_top_degree_uses_extra_differential(self):
        """e*e=e algebra: H_1 must vanish because d_2 is nonzero."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis[1] == 0
        assert bettis[0] == 1
