"""Tests for additive combinatorics operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive._models import (
    AdditiveEnergyRequest,
    AdditiveEnergyResult,
    DirectSumPredicateRequest,
    DirectSumPredicateResult,
    FiniteIntegerSet,
    RepresentationProfileRequest,
    RepresentationProfileResult,
    SumsetCardinalityRequest,
    SumsetCardinalityResult,
)
from jacobian.math.combinatorics.additive.operations import (
    additive_energy,
    direct_sum_predicate,
    representation_profile,
    sumset_cardinality,
)
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet as CanonicalFiniteIntegerSet,
)
from jacobian.math.combinatorics.finite_structures.sets.operations import (
    set_intersection,
)


def _run_representation(
    request: RepresentationProfileRequest,
) -> RepresentationProfileResult:
    return representation_profile(request.left, request.right)


def _run_energy(request: AdditiveEnergyRequest) -> AdditiveEnergyResult:
    return additive_energy(request.left, request.right)


def _run_sumset(request: SumsetCardinalityRequest) -> SumsetCardinalityResult:
    return sumset_cardinality(request.left, request.right)


def _run_direct_sum(request: DirectSumPredicateRequest) -> DirectSumPredicateResult:
    return direct_sum_predicate(request.modulus, request.left, request.right)


class TestRepresentationProfile:
    def test_finite_set_value_is_shared_with_the_finite_sets_owner(self) -> None:
        value = CanonicalFiniteIntegerSet(elements=("1", "2"))

        assert FiniteIntegerSet is CanonicalFiniteIntegerSet
        request = RepresentationProfileRequest(left=value, right=value)
        assert request.left is value
        assert request.right is value

        serialized_request = RepresentationProfileRequest.model_validate(
            {
                "left": value.model_dump(mode="json"),
                "right": value.model_dump(mode="json"),
            }
        )
        assert serialized_request.left == value
        assert serialized_request.right == value

    def test_two_by_two(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("1", "2")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = _run_representation(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"4": 1, "5": 2, "6": 1}

    def test_empty_set(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=("1", "2")),
        )
        result = _run_representation(req)
        assert result.entries == ()

    def test_self_sum(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "1", "2")),
        )
        result = _run_representation(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"0": 1, "1": 2, "2": 3, "3": 2, "4": 1}

    def test_negative_integers(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("-2", "-1")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = _run_representation(req)
        entries = tuple((entry.sum, entry.multiplicity) for entry in result.entries)
        assert entries == (("1", 1), ("2", 2), ("3", 1))

    def test_sums_sorted_and_unique(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = _run_representation(req)
        assert tuple(entry.sum for entry in result.entries) == (
            "-7",
            "-5",
            "-2",
            "0",
            "2",
            "3",
            "5",
            "7",
            "12",
        )

        assert tuple(entry.sum for entry in result.entries) == tuple(
            sorted({e.sum for e in result.entries}, key=parse_canonical_integer)
        )


class TestAdditiveEnergy:
    def test_two_by_two(self) -> None:
        req = AdditiveEnergyRequest(
            left=FiniteIntegerSet(elements=("1", "2")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = _run_energy(req)
        assert result.energy == 6  # 1^2 + 2^2 + 1^2

    def test_equal_sets(self) -> None:
        req = AdditiveEnergyRequest(
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = _run_energy(req)
        # A+A = {0,1,2}, r(0)=1, r(1)=2, r(2)=1 => E = 1+4+1 = 6
        assert result.energy == 6


class TestSumsetCardinality:
    def test_three_plus_two(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = _run_sumset(req)
        assert result.cardinality == 5
        assert result.support == FiniteIntegerSet(elements=("0", "1", "2", "3", "4"))

    def test_disjoint(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("10",)),
            right=FiniteIntegerSet(elements=("20",)),
        )
        result = _run_sumset(req)
        assert result.cardinality == 1

    def test_sumset_support_matches_profile(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = _run_sumset(req)
        assert result.support == FiniteIntegerSet(
            elements=("-7", "-5", "-2", "0", "2", "3", "5", "7", "12")
        )

    def test_support_composes_as_a_canonical_finite_set(self) -> None:
        result = sumset_cardinality(
            FiniteIntegerSet(elements=("0", "1", "2")),
            FiniteIntegerSet(elements=("0", "2")),
        )

        serialized_support = result.support.model_dump(mode="json")
        support = CanonicalFiniteIntegerSet.model_validate(serialized_support)
        intersection = set_intersection(
            support, FiniteIntegerSet(elements=("1", "3", "8"))
        )
        profile = representation_profile(support, FiniteIntegerSet(elements=("0",)))

        assert intersection == FiniteIntegerSet(elements=("1", "3"))
        assert tuple(entry.sum for entry in profile.entries) == support.elements

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ((), ("1", "2"), ()),
            (("-3", "-1"), ("-2", "1"), ("-5", "-3", "-2", "0")),
            (("0", "1", "2"), ("0", "1", "2"), ("0", "1", "2", "3", "4")),
        ],
    )
    def test_support_preserves_degenerate_and_duplicate_producing_sumsets(
        self,
        left: tuple[str, ...],
        right: tuple[str, ...],
        expected: tuple[str, ...],
    ) -> None:
        result = sumset_cardinality(
            FiniteIntegerSet(elements=left), FiniteIntegerSet(elements=right)
        )

        assert result.support == FiniteIntegerSet(elements=expected)
        assert result.cardinality == len(expected)

    def test_maximum_admitted_cartesian_product_can_return_distinct_support(
        self,
    ) -> None:
        left = FiniteIntegerSet(elements=tuple(str(value) for value in range(256)))
        right = FiniteIntegerSet(
            elements=tuple(str(256 * value) for value in range(256))
        )

        result = sumset_cardinality(left, right)

        assert result.cardinality == 65_536
        assert result.support.elements == tuple(str(value) for value in range(65_536))

    def test_result_support_can_repeat_a_large_operand_component(self) -> None:
        """Result growth is governed by its carrier, not the source digit total."""
        large = "1" + "0" * 999
        left = FiniteIntegerSet(elements=(large,))
        right = FiniteIntegerSet(elements=tuple(str(value) for value in range(64)))

        result = sumset_cardinality(left, right)
        assert result.cardinality == 64
        assert result.support.elements[0] == large

    def test_result_rejects_cardinality_that_disagrees_with_canonical_support(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="cardinality must equal"):
            SumsetCardinalityResult(
                cardinality=1,
                support=FiniteIntegerSet(elements=("0", "1")),
            )

    def test_result_rejects_noncanonical_support_order(self) -> None:
        with pytest.raises(ValidationError, match="support must be sorted"):
            SumsetCardinalityResult(
                cardinality=2,
                support=FiniteIntegerSet(elements=("1", "0")),
            )


class TestDirectSumPredicate:
    def test_tiling_z4(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = _run_direct_sum(req)
        assert result.holds is True
        assert result.collisions == ()
        assert result.missing == ()

    def test_non_tiling_z4(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = _run_direct_sum(req)
        assert result.holds is False

    def test_z6_tiling(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=6,
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "3")),
        )
        result = _run_direct_sum(req)
        assert result.holds is True

    def test_empty_sets_in_z12_return_numeric_missing(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=12,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )
        result = _run_direct_sum(req)
        assert result.holds is False
        assert result.missing == tuple(str(value) for value in range(12))

    def test_rejects_oversized_missing_diagnostic_by_cardinality(self) -> None:
        request = DirectSumPredicateRequest(
            modulus=1_200_000,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )

        with pytest.raises(OperationDomainValidationError) as exc_info:
            _run_direct_sum(request)

        assert exc_info.value.errors()[0]["type"] == (
            "additive_combinatorics.direct_sum.result_cardinality_exceeded"
        )
