"""Tests for Hochschild complex operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.hochschild_complexes._models import (
    MAX_HOCHSCHILD_MATRIX_ENTRIES,
    AlgebraStructure,
    HochschildChainComplexRequest,
    HochschildChainComplexResult,
    HochschildDifferential,
    HochschildHomologyRequest,
    HochschildHomologyResult,
)
from jacobian.math.hochschild_complexes._operations import (
    compute_hochschild_chain_complex,
    compute_hochschild_homology,
)


def _coordinatewise_algebra(prime: int, dimension: int) -> AlgebraStructure:
    """The associative coordinatewise algebra GF(prime)^dimension."""
    return AlgebraStructure(
        prime=prime,
        dimension=dimension,
        structure_constants=tuple(
            tuple(
                tuple(1 if i == j == k else 0 for k in range(dimension))
                for j in range(dimension)
            )
            for i in range(dimension)
        ),
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

    def test_dense_elimination_budget_rejected(self):
        """GF(2)^7 at max_degree=4 passes the tensor budget but not the matrix budget."""
        alg = _coordinatewise_algebra(2, 7)
        assert alg.dimension ** (4 + 1) <= 20_000
        with pytest.raises(ValidationError, match="boundary-matrix"):
            HochschildHomologyRequest(algebra=alg, max_degree=4)
        with pytest.raises(ValidationError, match="boundary-matrix"):
            HochschildChainComplexRequest(algebra=alg, max_degree=4)

    def test_largest_admitted_homology_request(self):
        """The densest admitted elimination stays inside the entry budget."""
        alg = _coordinatewise_algebra(5, 5)
        request = HochschildHomologyRequest(algebra=alg, max_degree=3)
        assert MAX_HOCHSCHILD_MATRIX_ENTRIES >= alg.dimension**7 == 78_125
        result = compute_hochschild_homology(request)
        assert [group.betti for group in result.groups] == [1, 0, 0, 0]


class TestChainComplexSourceBinding:
    """The chain complex result must be bound to its retained source algebra."""

    def _swap_algebra(self) -> AlgebraStructure:
        return AlgebraStructure(
            prime=7,
            dimension=2,
            structure_constants=(((1, 0), (0, 1)), ((0, 1), (1, 0))),
        )

    def test_result_retains_and_replays_source(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=3)
        )
        assert result.algebra == algebra
        assert result.algebra_dimension == 2
        assert result.group_dimensions == (1, 2, 4, 8)
        HochschildChainComplexResult.model_validate(result.model_dump())

    def test_authored_payload_rejected(self):
        with pytest.raises(ValidationError):
            HochschildChainComplexResult(
                algebra_dimension=5,
                group_dimensions=(1, 2),
                differentials=(
                    HochschildDifferential(
                        degree=1, source_dim=2, target_dim=1, entries=((0, 0),)
                    ),
                ),
                prime=5,
            )

    def test_tampered_differential_entry_rejected(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        payload = result.model_dump(mode="json")
        original = payload["differentials"][1]["entries"][0][0]
        payload["differentials"][1]["entries"][0][0] = (original + 1) % 7
        with pytest.raises(ValidationError, match="exact bar differential"):
            HochschildChainComplexResult.model_validate(payload)

    def test_inconsistent_group_dimensions_rejected(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        payload = result.model_dump()
        payload["group_dimensions"] = (1, 3, 9)
        with pytest.raises(ValidationError, match="group_dimensions"):
            HochschildChainComplexResult.model_validate(payload)

    def test_mismatched_prime_rejected(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=1)
        )
        payload = result.model_dump()
        payload["prime"] = 5
        with pytest.raises(ValidationError, match="retained algebra"):
            HochschildChainComplexResult.model_validate(payload)


class TestHomologySourceBinding:
    def test_forged_groups_rejected(self):
        from pydantic import ValidationError

        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
        )
        genuine = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        payload = genuine.model_dump()
        payload["groups"] = [{"degree": 0, "betti": 99}]
        with pytest.raises(ValidationError, match="replay"):
            HochschildHomologyResult.model_validate(payload)

    def test_prime_mismatch_rejected(self):
        from pydantic import ValidationError

        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
        )
        genuine = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        payload = genuine.model_dump()
        payload["prime"] = 7
        with pytest.raises(ValidationError, match="prime"):
            HochschildHomologyResult.model_validate(payload)
