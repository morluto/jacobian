"""Tests for additive combinatorics operations."""

import pytest

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive._models import (
    AdditiveEnergyRequest,
    DirectSumPredicateRequest,
    FiniteIntegerSet,
    RepresentationProfileRequest,
    SumsetCardinalityRequest,
    _direct_sum_predicate_result_upper_bound,
)
from jacobian.math.combinatorics.additive._operations import (
    compute_additive_energy,
    compute_representation_profile,
    compute_sumset_cardinality,
    decide_direct_sum_predicate,
)
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet as CanonicalFiniteIntegerSet,
)


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
        result = compute_representation_profile(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"4": 1, "5": 2, "6": 1}

    def test_empty_set(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=("1", "2")),
        )
        result = compute_representation_profile(req)
        assert result.entries == ()

    def test_self_sum(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "1", "2")),
        )
        result = compute_representation_profile(req)
        entries = {e.sum: e.multiplicity for e in result.entries}
        assert entries == {"0": 1, "1": 2, "2": 3, "3": 2, "4": 1}

    def test_negative_integers(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("-2", "-1")),
            right=FiniteIntegerSet(elements=("3", "4")),
        )
        result = compute_representation_profile(req)
        entries = tuple((entry.sum, entry.multiplicity) for entry in result.entries)
        assert entries == (("1", 1), ("2", 2), ("3", 1))

    def test_sums_sorted_and_unique(self) -> None:
        req = RepresentationProfileRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = compute_representation_profile(req)
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
        result = compute_additive_energy(req)
        assert result.energy == 6  # 1^2 + 2^2 + 1^2

    def test_equal_sets(self) -> None:
        req = AdditiveEnergyRequest(
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = compute_additive_energy(req)
        # A+A = {0,1,2}, r(0)=1, r(1)=2, r(2)=1 => E = 1+4+1 = 6
        assert result.energy == 6


class TestSumsetCardinality:
    def test_three_plus_two(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = compute_sumset_cardinality(req)
        assert result.cardinality == 5
        assert result.support == ("0", "1", "2", "3", "4")

    def test_disjoint(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("10",)),
            right=FiniteIntegerSet(elements=("20",)),
        )
        result = compute_sumset_cardinality(req)
        assert result.cardinality == 1

    def test_sumset_support_matches_profile(self) -> None:
        req = SumsetCardinalityRequest(
            left=FiniteIntegerSet(elements=("7", "-2", "0")),
            right=FiniteIntegerSet(elements=("5", "0", "-5")),
        )
        result = compute_sumset_cardinality(req)
        assert result.support == (
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


class TestDirectSumPredicate:
    def test_tiling_z4(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "2")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is True
        assert result.collisions == ()
        assert result.missing == ()

    def test_non_tiling_z4(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=4,
            left=FiniteIntegerSet(elements=("0", "1")),
            right=FiniteIntegerSet(elements=("0", "1")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is False

    def test_z6_tiling(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=6,
            left=FiniteIntegerSet(elements=("0", "1", "2")),
            right=FiniteIntegerSet(elements=("0", "3")),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is True

    def test_empty_sets_in_z12_return_numeric_missing(self) -> None:
        req = DirectSumPredicateRequest(
            modulus=12,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )
        result = decide_direct_sum_predicate(req)
        assert result.holds is False
        assert result.missing == tuple(str(value) for value in range(12))

    def test_preflights_complete_missing_diagnostic_at_transport_boundary(self) -> None:
        """The nearly 10 MiB empty-source result remains serializable."""

        modulus = 1_048_576
        request = DirectSumPredicateRequest(
            modulus=modulus,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )

        payload = {
            "holds": False,
            "modulus": modulus,
            "representatives": [],
            "collisions": [],
            "missing": [str(value) for value in range(modulus)],
        }
        encoded = encode_strict_json(payload)
        assert len(encoded) <= CanonicalLimits().max_output_bytes
        assert _direct_sum_predicate_result_upper_bound(modulus, 0) >= len(encoded)
        assert request.modulus == modulus

    def test_rejects_oversized_missing_diagnostic_before_kernel_execution(self) -> None:
        request = DirectSumPredicateRequest(
            modulus=1_200_000,
            left=FiniteIntegerSet(elements=()),
            right=FiniteIntegerSet(elements=()),
        )

        with pytest.raises(OperationDomainValidationError) as exc_info:
            decide_direct_sum_predicate(request)

        assert exc_info.value.errors()[0]["type"] == (
            "additive_combinatorics.direct_sum_result_transport_exceeded"
        )
