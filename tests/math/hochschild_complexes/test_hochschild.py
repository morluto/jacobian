"""Tests for Hochschild complex operations."""

from __future__ import annotations

from contextlib import contextmanager

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
    verify_hochschild_chain_complex_result,
    verify_hochschild_homology_result,
)
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
)


@contextmanager
def _validation_error(code: str):
    with pytest.raises(ValidationError) as error:
        yield
    assert error.value.errors()[0]["type"] == code


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
        augmentation=tuple(1 if i == 0 else 0 for i in range(dimension)),
    )


class TestHochschildChainComplex:
    """Test Hochschild chain complex computation."""

    def test_one_dim_algebra(self):
        """Chain complex of a 1D algebra over GF(5)."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
            augmentation=(1,),
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
            augmentation=(1, 1),
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
            augmentation=(1, 5 - 1),
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
            augmentation=(1,),
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
            augmentation=(1, 1),
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
            augmentation=(0,),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        # With zero multiplication, the differential vanishes
        assert result.groups[0].betti >= 0


class TestHochschildAdmissionAndTopDegree:
    def test_composite_prime_rejected(self):
        with _validation_error("hochschild_complex.prime"):
            AlgebraStructure(
                prime=4,
                dimension=1,
                structure_constants=(((1,),),),
                augmentation=(1,),
            )

    def test_non_associative_rejected(self):
        """[e0,e1]=e0 style left-zero multiplication fails associativity."""
        c = (
            ((0, 0), (0, 0)),
            ((1, 0), (0, 0)),
        )
        with _validation_error("hochschild_complex.associativity"):
            AlgebraStructure(
                prime=5,
                dimension=2,
                structure_constants=c,
                augmentation=(0, 0),
            )

    def test_top_degree_uses_extra_differential(self):
        """e*e=e algebra: H_1 must vanish because d_2 is nonzero."""
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
            augmentation=(1,),
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
        with _validation_error("hochschild_complex.matrix_budget"):
            HochschildHomologyRequest(algebra=alg, max_degree=4)
        with _validation_error("hochschild_complex.matrix_budget"):
            HochschildChainComplexRequest(algebra=alg, max_degree=4)

    def test_largest_admitted_homology_request(self):
        """The densest admitted elimination stays inside the entry budget."""
        alg = _coordinatewise_algebra(5, 5)
        request = HochschildHomologyRequest(algebra=alg, max_degree=3)
        assert MAX_HOCHSCHILD_MATRIX_ENTRIES >= alg.dimension**7 == 78_125
        result = compute_hochschild_homology(request)
        assert [group.betti for group in result.groups] == [1, 0, 0, 0]

    def test_nine_dim_coordinatewise_algebra_admitted(self):
        """GF(5)^9 with max_degree=1 fits every derived budget and must be admitted.

        Regression: a fixed dimension<=8 ceiling used to reject it even though
        its 729 structure constants, 81 highest tensor elements, 9x81 largest
        homology boundary, and 9-wide chain-complex matrix are all inside the
        declared budgets.
        """
        alg = _coordinatewise_algebra(5, 9)
        assert alg.dimension**3 == 729
        homology = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=1)
        )
        # GF(5)^9 is separable with the projection augmentation, so
        # HH_0 = K and all higher groups vanish.
        assert [group.betti for group in homology.groups] == [1, 0]
        complex_result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=alg, max_degree=1)
        )
        assert complex_result.group_dimensions == (1, 9)
        # The accepted payload round-trips through its own validator.
        HochschildChainComplexResult.model_validate(complex_result.model_dump())

    def test_associativity_admission_budget_rejected(self):
        """Dimensions whose 2*n^5 associativity walk exceeds the work budget fail."""
        from jacobian.math.hochschild_complexes._models import (
            MAX_ASSOCIATIVITY_DOT_STEPS,
        )

        boundary = 25
        assert 2 * boundary**5 <= MAX_ASSOCIATIVITY_DOT_STEPS
        assert 2 * (boundary + 1) ** 5 > MAX_ASSOCIATIVITY_DOT_STEPS
        with _validation_error("hochschild_complex.associativity_budget"):
            _coordinatewise_algebra(2, boundary + 1)

    def test_structure_input_budget_rejected(self):
        """Multiplication tables beyond the dense-payload entry budget fail."""
        from jacobian.math.hochschild_complexes._models import (
            MAX_STRUCTURE_CONSTANT_ENTRIES,
        )

        oversized = 51
        assert oversized**3 > MAX_STRUCTURE_CONSTANT_ENTRIES
        with _validation_error("hochschild_complex.input_budget"):
            _coordinatewise_algebra(2, oversized)

    def test_request_budgets_bind_above_the_old_dimension_ceiling(self):
        """Larger admitted algebras still face the per-request envelopes."""
        with _validation_error("hochschild_complex.tensor_budget"):
            HochschildHomologyRequest(
                algebra=_coordinatewise_algebra(2, 10), max_degree=4
            )


class TestChainComplexSourceBinding:
    """The chain complex result must be bound to its retained source algebra."""

    def _swap_algebra(self) -> AlgebraStructure:
        return AlgebraStructure(
            prime=7,
            dimension=2,
            structure_constants=(((1, 0), (0, 1)), ((0, 1), (1, 0))),
            augmentation=(1, 1),
        )

    def test_result_retains_and_replays_source(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=3)
        )
        assert result.algebra == algebra
        assert result.algebra_dimension == 2
        assert result.group_dimensions == (1, 2, 4, 8)
        assert verify_hochschild_chain_complex_result(result)
        HochschildChainComplexResult.model_validate(result.model_dump())

    def test_authored_payload_rejected(self):
        with _validation_error("missing"):
            HochschildChainComplexResult(
                algebra_dimension=5,
                group_dimensions=(1, 2),
                differentials=(
                    HochschildDifferential(
                        degree=1,
                        matrix=PrimeFieldMatrix(prime=5, entries=((0, 0),), columns=2),
                    ),
                ),
                prime=5,
            )

    def test_tampered_differential_entry_requires_explicit_verifier(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        payload = result.model_dump(mode="json")
        original = payload["differentials"][1]["matrix"]["entries"][0][0]
        payload["differentials"][1]["matrix"]["entries"][0][0] = (original + 1) % 7
        supplied = HochschildChainComplexResult.model_validate(payload)
        assert not verify_hochschild_chain_complex_result(supplied)

    def test_inconsistent_group_dimensions_rejected(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        payload = result.model_dump()
        payload["group_dimensions"] = (1, 3, 9)
        with _validation_error("hochschild_complex.group_dimensions"):
            HochschildChainComplexResult.model_validate(payload)

    def test_mismatched_prime_rejected(self):
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=1)
        )
        payload = result.model_dump()
        payload["prime"] = 5
        with _validation_error("hochschild_complex.algebra_binding"):
            HochschildChainComplexResult.model_validate(payload)

    def test_mismatched_differential_prime_rejected(self):
        """A differential over another prime must be rejected, not reinterpreted.

        The exact GF(5) entries stay canonical residues of the larger GF(7),
        so only an explicit prime-binding check can stop the payload from
        silently switching fields inside rank/RREF/nullspace consumers.
        """
        algebra = _dual_numbers(5)
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=1)
        )
        assert result.differentials[0].matrix.entries != ((0,),)
        payload = result.model_dump()
        payload["differentials"][0]["matrix"]["prime"] = 7
        with _validation_error("hochschild_complex.differential_prime"):
            HochschildChainComplexResult.model_validate(payload)

    def test_all_zero_differential_foreign_prime_rejected(self):
        """The review's example: an all-zero GF(7) differential for a GF(5) algebra."""
        zero_algebra = AlgebraStructure(
            prime=5,
            dimension=2,
            structure_constants=(
                ((0, 0), (0, 0)),
                ((0, 0), (0, 0)),
            ),
            augmentation=(0, 0),
        )
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=zero_algebra, max_degree=1)
        )
        assert result.differentials[0].matrix.entries == ((0, 0),)
        payload = result.model_dump()
        payload["differentials"][0]["matrix"]["prime"] = 7
        with _validation_error("hochschild_complex.differential_prime"):
            HochschildChainComplexResult.model_validate(payload)

    def test_matching_differential_prime_accepted(self):
        """Round-tripped differentials carrying the algebra prime still validate."""
        algebra = self._swap_algebra()
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        assert all(
            differential.matrix.prime == algebra.prime
            for differential in result.differentials
        )
        HochschildChainComplexResult.model_validate(result.model_dump())


class TestHomologySourceBinding:
    def test_forged_groups_require_explicit_verifier(self):
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
            augmentation=(1,),
        )
        genuine = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        assert verify_hochschild_homology_result(genuine)
        payload = genuine.model_dump()
        payload["groups"][1]["betti"] = 1
        supplied = HochschildHomologyResult.model_validate(payload)
        assert not verify_hochschild_homology_result(supplied)

    def test_prime_mismatch_rejected(self):
        alg = AlgebraStructure(
            prime=5,
            dimension=1,
            structure_constants=(((1,),),),
            augmentation=(1,),
        )
        genuine = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=alg, max_degree=2)
        )
        payload = genuine.model_dump()
        payload["prime"] = 7
        with _validation_error("hochschild_complex.prime_binding"):
            HochschildHomologyResult.model_validate(payload)


def _dual_numbers(prime: int) -> AlgebraStructure:
    """GF(prime)[x]/(x^2) with basis (1, x) and the standard augmentation."""
    return AlgebraStructure(
        prime=prime,
        dimension=2,
        structure_constants=(
            ((1, 0), (0, 1)),
            ((0, 1), (0, 0)),
        ),
        augmentation=(1, 0),
    )


class TestAugmentationEndpointFaces:
    """The trivial module acts through epsilon, so both endpoint faces count."""

    def test_dual_numbers_hh_is_one_in_every_degree(self):
        """HH_n(GF(p)[x]/(x^2), K) = K for all n; adjacent-only would give H_1 = 0."""
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=_dual_numbers(5), max_degree=4)
        )
        assert [group.betti for group in result.groups] == [1, 1, 1, 1, 1]

    def test_zero_augmentation_reduces_to_adjacent_faces(self):
        """epsilon = 0 kills both endpoint faces, leaving interior multiplication."""
        zeroed = AlgebraStructure(
            prime=5,
            dimension=2,
            structure_constants=(
                ((1, 0), (0, 1)),
                ((0, 1), (0, 0)),
            ),
            augmentation=(0, 0),
        )
        result = compute_hochschild_homology(
            HochschildHomologyRequest(algebra=zeroed, max_degree=3)
        )
        # With no augmentation action the image of d_2 is all of A
        # (1*1 = 1 spans), so H_1 vanishes - the pre-fix behaviour.
        bettis = {group.degree: group.betti for group in result.groups}
        assert bettis[1] == 0

    def test_differential_squares_to_zero(self):
        """d^2 = 0 for consecutive degrees on two different augmented algebras."""
        from jacobian.math.hochschild_complexes._bar import (
            bar_differential_entries,
        )

        algebras = [
            _dual_numbers(5),
            AlgebraStructure(
                prime=7,
                dimension=2,
                structure_constants=(
                    ((1, 0), (0, 1)),
                    ((0, 1), (1, 0)),
                ),
                augmentation=(1, 6),
            ),
        ]
        for algebra in algebras:
            for degree in range(1, 5):
                d_mid = bar_differential_entries(  # C_degree -> C_{degree-1}
                    algebra.structure_constants,
                    algebra.prime,
                    degree,
                    algebra.augmentation,
                )
                d_high = bar_differential_entries(  # C_{degree+1} -> C_degree
                    algebra.structure_constants,
                    algebra.prime,
                    degree + 1,
                    algebra.augmentation,
                )
                composition = [
                    [
                        sum(d_mid[i][k] * d_high[k][j] for k in range(len(d_mid[i])))
                        % algebra.prime
                        for j in range(len(d_high[0]))
                    ]
                    for i in range(len(d_mid))
                ]
                assert all(value == 0 for row in composition for value in row)

    def test_non_multiplicative_augmentation_rejected(self):
        """An augmentation that is not an algebra map must fail admission."""
        with _validation_error("hochschild_complex.augmentation_homomorphism"):
            AlgebraStructure(
                prime=5,
                dimension=2,
                structure_constants=(
                    ((1, 0), (0, 1)),
                    ((0, 1), (0, 0)),
                ),
                augmentation=(1, 1),
            )

    def test_noncanonical_and_mismatched_augmentation_rejected(self):
        dual = _dual_numbers(5)
        with _validation_error("hochschild_complex.canonical_residues"):
            AlgebraStructure(
                prime=5,
                dimension=2,
                structure_constants=dual.structure_constants,
                augmentation=(1, 5),
            )
        with _validation_error("hochschild_complex.augmentation_shape"):
            AlgebraStructure(
                prime=5,
                dimension=2,
                structure_constants=dual.structure_constants,
                augmentation=(1,),
            )

    def test_result_verifier_uses_retained_augmentation(self):
        """Tampering with the retained augmentation invalidates entries."""
        from jacobian.math.hochschild_complexes._models import (
            HochschildChainComplexResult,
        )

        algebra = _dual_numbers(5)
        result = compute_hochschild_chain_complex(
            HochschildChainComplexRequest(algebra=algebra, max_degree=2)
        )
        payload = result.model_dump()
        payload["algebra"]["augmentation"] = [0, 0]
        supplied = HochschildChainComplexResult.model_validate(payload)
        assert not verify_hochschild_chain_complex_result(supplied)
