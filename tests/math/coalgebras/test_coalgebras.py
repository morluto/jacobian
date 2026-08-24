"""Tests for coalgebra operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.coalgebras._models import (
    GROUP_LIKE_SCAN_WORK_BUDGET,
    MAX_PRIME_DIGITS,
    MAX_TENSOR_ENTRIES,
    Coalgebra,
    ComultiplicationRequest,
    ComultiplicationResult,
    CounitRequest,
    CounitResult,
    GroupLikeElementsRequest,
    GroupLikeElementsResult,
    group_like_scan_work,
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

    def test_scan_work_budget_rejected(self):
        """Requests whose derived scan work exceeds the budget are rejected."""
        oversized = _direct_sum_group_like_coalgebra(12)
        assert group_like_scan_work(2, 12) > GROUP_LIKE_SCAN_WORK_BUDGET
        with pytest.raises(ValidationError, match="scan work exceeds"):
            GroupLikeElementsRequest(coalgebra=oversized)

        large_prime_squared = Coalgebra(
            prime=9973,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert group_like_scan_work(9973, 2) > GROUP_LIKE_SCAN_WORK_BUDGET
        with pytest.raises(ValidationError, match="scan work exceeds"):
            GroupLikeElementsRequest(coalgebra=large_prime_squared)

    def test_within_scan_work_budget_admitted(self):
        """An element space whose scan work fits the budget enumerates."""
        ca = Coalgebra(
            prime=251,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert (
            group_like_scan_work(ca.prime, ca.dimension) <= GROUP_LIKE_SCAN_WORK_BUDGET
        )
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

    def test_detached_result_reapplies_scan_work_budget(self):
        """A serialized result validates its coalgebra as a plain Coalgebra,
        so the replay must reapply the derived scan-work admission itself."""
        ca = Coalgebra(
            prime=9973,
            dimension=2,
            comultiplication=(
                ((1, 0), (0, 0)),
                ((0, 0), (0, 1)),
            ),
            counit=(1, 1),
        )
        assert (
            group_like_scan_work(ca.prime, ca.dimension) > GROUP_LIKE_SCAN_WORK_BUDGET
        )
        with pytest.raises(ValidationError, match="scan work exceeds"):
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


class TestScanWorkBeforeReplay:
    def test_oversized_direct_result_rejected_before_enumeration(self):
        """A serialized result with an over-budget scan fails fast."""
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
        assert (
            group_like_scan_work(big.prime, big.dimension) > GROUP_LIKE_SCAN_WORK_BUDGET
        )
        started = time.monotonic()
        with pytest.raises(ValidationError, match="scan work exceeds"):
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
    """Dimension admission derives from tensor size and scan work."""

    def test_nine_dim_gf2_direct_sum_admitted(self):
        """729 tensor entries and ~2*10**5 scan-work units fit both budgets."""
        ca = _direct_sum_group_like_coalgebra(9, prime=2)
        assert ca.dimension**3 == 729
        assert (
            group_like_scan_work(ca.prime, ca.dimension) <= GROUP_LIKE_SCAN_WORK_BUDGET
        )

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

    def test_large_prime_nine_dim_rejected_by_scan_work_budget(self):
        """A 9-dim GF(13) coalgebra fits the tensor budget but reconstructing
        its surviving candidates exceeds the scan-work budget."""
        ca = _direct_sum_group_like_coalgebra(9, prime=13)
        assert ca.dimension**3 == 729 <= MAX_TENSOR_ENTRIES
        with pytest.raises(ValidationError, match="scan work exceeds"):
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


class TestScanWorkBoundary:
    """Admission bounds combined kernel-plus-replay work, not just the
    candidate count."""

    def test_largest_admitted_gf2_boundary_completes_quickly(self):
        """The admitted boundary (11-dim GF(2) direct sum) finishes fast even
        though kernel and replay each scan its whole element space."""
        import time

        ca = _direct_sum_group_like_coalgebra(11)
        assert (
            group_like_scan_work(ca.prime, ca.dimension) <= GROUP_LIKE_SCAN_WORK_BUDGET
        )
        started = time.monotonic()
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        elapsed = time.monotonic() - started
        assert result.count == 11
        assert {tuple(e.coefficients) for e in result.elements} == {
            tuple(1 if i == j else 0 for i in range(11)) for j in range(11)
        }
        assert elapsed < 5

    def test_first_rejected_boundary_names_the_budget(self):
        """12-dim GF(2) direct sum: 4096 candidates whose reconstruction
        exceeds the documented scan-work budget are rejected up front."""
        ca = _direct_sum_group_like_coalgebra(12)
        assert (
            group_like_scan_work(ca.prime, ca.dimension) > GROUP_LIKE_SCAN_WORK_BUDGET
        )
        with pytest.raises(ValidationError, match="scan work exceeds"):
            GroupLikeElementsRequest(coalgebra=ca)

    def test_reported_sixteen_dim_request_typed_rejected(self, monkeypatch):
        """The originally reported 16-dim GF(2) direct-sum request pays
        roughly 152M reconstruction units per pass (kernel plus replay),
        far above the budget, so it fails admission without enumerating.

        Rejection is proven structurally: the predicted scan work exceeds
        the budget, and the typed admission error (not a completed scan)
        is what fires. The group-like kernel is patched to fail if
        admission ever triggers enumeration. No wall-clock bound:
        construction legitimately pays the O(dimension^4) coalgebra-axiom
        verification, which varies several fold on loaded CI runners.
        """
        import jacobian.math.coalgebras._operations as coalgebra_operations

        def forbid_enumeration(coalgebra):
            raise AssertionError("admission must reject before enumeration")

        monkeypatch.setattr(
            coalgebra_operations, "_group_like_coefficients", forbid_enumeration
        )
        ca = _direct_sum_group_like_coalgebra(16)
        assert (
            group_like_scan_work(ca.prime, ca.dimension) > GROUP_LIKE_SCAN_WORK_BUDGET
        )
        with pytest.raises(ValidationError, match="scan work exceeds"):
            GroupLikeElementsRequest(coalgebra=ca)


class TestNestedModulusPrevalidation:
    """The raw nested matrix modulus is checked before PrimeFieldMatrix
    construction runs its primality test."""

    def _payload(self, matrix_prime: int) -> dict:
        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        return {
            "coalgebra": ca.model_dump(),
            "element_index": 0,
            "matrix": {"prime": matrix_prime, "entries": [[1]], "columns": 1},
            "dimension": 1,
        }

    def test_huge_nested_modulus_rejected_before_primality_test(self):
        """A multi-thousand-digit nested modulus is rejected by digit
        admission, never reaching the shared type's unbounded primality
        test."""
        import time

        payload = self._payload(10**6000 + 4567)
        started = time.monotonic()
        with pytest.raises(ValidationError, match="digit admission bound"):
            ComultiplicationResult.model_validate(payload)
        assert time.monotonic() - started < 5

    def test_foreign_nested_modulus_rejected_before_matrix_construction(self):
        """A 63-digit composite passes the digit budget but would fail the
        shared type's primality test; the field-mismatch message proves the
        raw-modulus check ran first."""
        composite = 2 * 10**62
        with pytest.raises(
            ValidationError, match="matrix prime must match the retained"
        ):
            ComultiplicationResult.model_validate(self._payload(composite))

    def test_matching_nested_modulus_still_round_trips(self):
        from jacobian.math.coalgebras._operations import compute_comultiplication

        ca = Coalgebra(
            prime=5,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        result = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=0)
        )
        assert ComultiplicationResult.model_validate(result.model_dump()) == result


class TestPrimeDigitAdmission:
    """Field admission derives from characteristic digit length, not a fixed
    magnitude ceiling."""

    def test_one_dimensional_gf10007_coalgebra_admitted(self):
        """The one-entry GF(10007) coalgebra is admissible end to end."""
        ca = Coalgebra(
            prime=10007,
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        comult = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=0)
        )
        assert comult.matrix.entries == ((1,),)
        counit = compute_counit(CounitRequest(coalgebra=ca, element_index=0))
        assert counit.value == 1
        result = find_group_like_elements(GroupLikeElementsRequest(coalgebra=ca))
        assert result.count == 1
        assert result.elements[0].coefficients == (1,)

    def test_multithousand_digit_prime_rejected_before_primality_test(self):
        import time

        started = time.monotonic()
        with pytest.raises(ValidationError, match="digit admission bound"):
            Coalgebra(
                prime=10**6000 + 4567,
                dimension=1,
                comultiplication=(((1,),),),
                counit=(1,),
            )
        assert time.monotonic() - started < 5

    def test_digit_boundary(self):
        """A 65-digit characteristic is rejected while a full-budget
        64-digit prime is admitted."""
        with pytest.raises(ValidationError, match="digit admission bound"):
            Coalgebra(
                prime=10**MAX_PRIME_DIGITS,
                dimension=1,
                comultiplication=(((1,),),),
                counit=(1,),
            )
        from sympy import nextprime

        ca = Coalgebra(
            prime=nextprime(10 ** (MAX_PRIME_DIGITS - 1)),
            dimension=1,
            comultiplication=(((1,),),),
            counit=(1,),
        )
        comult = compute_comultiplication(
            ComultiplicationRequest(coalgebra=ca, element_index=0)
        )
        assert comult.matrix.entries == ((1,),)
