"""Tests for coalgebra operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.coalgebras._models import (
    GROUP_LIKE_ENUMERATION_BUDGET,
    MAX_TENSOR_ENTRIES,
    Coalgebra,
    ComultiplicationRequest,
    ComultiplicationResult,
    CounitRequest,
    CounitResult,
    GroupLikeElementsRequest,
    GroupLikeElementsResult,
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
        assert result.matrix.entries[0][0] == 1

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
        assert result.matrix.entries[0][0] == 1
        assert result.matrix.entries[1][1] == 0


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
        assert ca.prime**ca.dimension > GROUP_LIKE_ENUMERATION_BUDGET
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
        assert ca.prime**ca.dimension <= GROUP_LIKE_ENUMERATION_BUDGET
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        # Both basis elements are group-like in this direct-sum coalgebra.
        assert result.count == 2
        found = {tuple(e.coefficients) for e in result.elements}
        assert found == {(1, 0), (0, 1)}


class TestSourceBoundResults:
    """Results retain their coalgebra and replay the defining relations."""

    def _two_dim_coalgebra(self) -> Coalgebra:
        return Coalgebra(
            prime=5,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )

    def test_results_revalidate_round_trip(self):
        ca = self._two_dim_coalgebra()
        comult = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=1)
        )
        counit = compute_counit(CounitRequest(coalgebra=ca, element_index=0))
        group_like = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert ComultiplicationResult.model_validate(comult.model_dump()) == comult
        assert CounitResult.model_validate(counit.model_dump()) == counit
        assert (
            GroupLikeElementsResult.model_validate(group_like.model_dump())
            == group_like
        )

    def test_detached_empty_group_like_result_is_rejected(self):
        with pytest.raises(ValidationError):
            GroupLikeElementsResult(elements=(), count=0)

    def test_forged_group_like_set_is_rejected(self):
        """A nonempty coalgebra cannot validate an empty enumeration."""
        ca = self._two_dim_coalgebra()
        with pytest.raises(ValueError, match="exact group-like set"):
            GroupLikeElementsResult(coalgebra=ca, elements=(), count=0)

    def test_detached_result_reapplies_enumeration_budget(self):
        """A serialized result validates its coalgebra as a plain Coalgebra,
        so the replay must reapply prime**dimension admission itself."""
        ca = Coalgebra(
            prime=9973,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert ca.prime**ca.dimension > GROUP_LIKE_ENUMERATION_BUDGET
        with pytest.raises(ValidationError, match="enumeration requires"):
            GroupLikeElementsResult(coalgebra=ca, elements=(), count=0)

    def test_forged_comultiplication_coefficients_are_rejected(self):
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        ca = self._two_dim_coalgebra()
        with pytest.raises(ValueError, match="exact comultiplication"):
            ComultiplicationResult(
                coalgebra=ca,
                element_index=0,
                matrix=PrimeFieldMatrix(
                    prime=ca.prime,
                    entries=((4, 0), (0, 0)),
                    columns=2,
                ),
                dimension=2,
            )

    def test_forged_counit_value_is_rejected(self):
        ca = self._two_dim_coalgebra()
        with pytest.raises(ValueError, match="exact counit"):
            CounitResult(coalgebra=ca, element_index=0, value=3)

    def test_result_from_other_coalgebra_is_rejected(self):
        ca = self._two_dim_coalgebra()
        other = Coalgebra(
            prime=5,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 1), (1, 0)),
            ),
            counit=(1, 0),
        )
        result = compute_counit(CounitRequest(coalgebra=ca, element_index=1))
        with pytest.raises(ValueError, match="exact counit"):
            CounitResult(
                coalgebra=other,
                element_index=result.element_index,
                value=result.value,
            )


class TestEnumerationBudgetBeforeReplay:
    def test_oversized_direct_result_rejected_before_enumeration(self):
        """A serialized result with an unbounded coalgebra fails fast."""
        import time

        big = Coalgebra(
            prime=9973,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert big.prime**big.dimension > GROUP_LIKE_ENUMERATION_BUDGET
        started = time.monotonic()
        with pytest.raises(ValidationError, match="enumeration requires"):
            GroupLikeElementsResult(coalgebra=big, elements=(), count=0)
        assert time.monotonic() - started < 5


class TestCanonicalResidues:
    def test_noncanonical_structure_constant_rejected(self):
        with pytest.raises(ValidationError, match="canonical residues"):
            Coalgebra(
                prime=2,
                dimension=1,
                comultiplication=(((3,),),),
                counit=(1,),
            )

    def test_noncanonical_counit_rejected(self):
        with pytest.raises(ValidationError, match="canonical residues"):
            Coalgebra(
                prime=3,
                dimension=1,
                comultiplication=(((1,),),),
                counit=(5,),
            )


def _direct_sum_group_like_coalgebra(n: int, prime: int = 2) -> Coalgebra:
    """The n-fold direct sum of trivial coalgebras: Delta(c_i) = c_i (x) c_i.

    Every basis element is group-like with counit 1, so the tensor carries
    exactly n^3 structure constants and the element space has prime**n
    candidates.
    """
    return Coalgebra(
        prime=prime,
        dimension=n,
        comultiplication=tuple(
            tuple(tuple(1 if j == i == k else 0 for k in range(n)) for j in range(n))
            for i in range(n)
        ),
        counit=(1,) * n,
    )


class TestDerivedDimensionAdmission:
    """Dimension admission derives from tensor size and enumeration work."""

    def test_nine_dim_gf2_direct_sum_admitted(self):
        """729 tensor entries and 512 group-like candidates fit both budgets."""
        ca = _direct_sum_group_like_coalgebra(9, prime=2)
        assert ca.dimension**3 == 729
        assert ca.prime**ca.dimension == 512

        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 9
        assert {tuple(e.coefficients) for e in result.elements} == {
            tuple(1 if i == j else 0 for i in range(9)) for j in range(9)
        }

        comult = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=8)
        )
        assert comult.matrix.entries[8][8] == 1

        counit = compute_counit(CounitRequest(coalgebra=ca, element_index=8))
        assert counit.value == 1

    def test_boundary_dimension_sixteen_admitted(self):
        """16^3 = 4096 entries sits exactly on the derived tensor budget."""
        ca = _direct_sum_group_like_coalgebra(16)
        assert ca.dimension**3 == MAX_TENSOR_ENTRIES
        result = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=15)
        )
        assert result.matrix.entries[15][15] == 1

    def test_above_tensor_budget_rejected(self):
        """A 17-dim tensor would carry 4913 structure constants and is rejected."""
        with pytest.raises(ValidationError, match="structure constants"):
            _direct_sum_group_like_coalgebra(17)

    def test_large_prime_nine_dim_rejected_by_enumeration_budget(self):
        """A 9-dim GF(13) coalgebra fits the tensor budget but its 13^9
        candidate space exceeds the enumeration budget."""
        ca = _direct_sum_group_like_coalgebra(9, prime=13)
        assert ca.dimension**3 == 729 <= MAX_TENSOR_ENTRIES
        with pytest.raises(ValidationError, match="enumeration requires"):
            GroupLikeElementsRequest(coalgebra=ca)


class TestCounitOperationRemovalFromCatalog:
    """epsilon(c_i) is a deterministic projection of retained data, so it
    stays native-only and out of the declared public operations."""

    def test_counit_operation_is_not_declared(self):
        from jacobian.math.coalgebras._tools import COALGEBRA_OPERATIONS

        assert {tool.operation_id for tool in COALGEBRA_OPERATIONS} == {
            "coalgebra.comultiplication.compute",
            "coalgebra.group_like_elements.compute",
        }

    def test_native_counit_kernel_remains_available(self):
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        result = compute_counit(CounitRequest(coalgebra=ca, element_index=0))
        assert result.value == 1
