"""Contract tests for exact finite periodic congruence unions."""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import cast

import pytest
from tests.math.number_theory._validation import expect_validation

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic import (
    compute_periodic_congruence_union_measure,
    compute_periodic_congruence_union_profile,
    normalize_periodic_source,
)
from jacobian.math.number_theory._periodic_kernel import (
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    MAX_INTERSECTION_MERGES,
    MAX_INTERSECTION_STATES,
    MAX_MATERIALIZED_RESIDUES,
    MAX_PERIOD_LIFT_WORK,
    MAX_PERIOD_SCAN,
    MAX_PERIODIC_FAMILY_SIZE,
    MAX_PERIODIC_INTEGER_DIGITS,
    MAX_PERIODIC_RESULT_BYTES,
    MAX_PERIODIC_SOURCE_ROWS,
    MAX_SPARSE_LIFTED_ROWS,
    PERIODIC_EXECUTION_PASSES_PER_CALL,
    PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES,
    PeriodicCongruenceSubsetInput,
    PeriodicCongruenceUnionMeasureResult,
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
    PeriodicCongruenceUnionRequest,
)
from jacobian.math.number_theory.operations import (
    periodic_congruence_union_measure as native_periodic_measure,
)
from jacobian.math.number_theory.operations import (
    periodic_congruence_union_profile as native_periodic_profile,
)


def _measure(payload: dict[str, object]) -> PeriodicCongruenceUnionMeasureResult:
    request = PeriodicCongruenceUnionRequest.model_validate(payload)
    return compute_periodic_congruence_union_measure(request)


def _profile(payload: dict[str, object]) -> PeriodicCongruenceUnionProfileResult:
    request = PeriodicCongruenceUnionProfileRequest.model_validate(payload)
    return compute_periodic_congruence_union_profile(request)


def test_native_periodic_operations_accept_canonical_source() -> None:
    request = PeriodicCongruenceUnionRequest.model_validate(
        {
            "subsets": [{"modulus": "4", "residues": ["1"]}],
            "complement": False,
        }
    )
    source = normalize_periodic_source(request)

    measure = native_periodic_measure(source)
    profile = native_periodic_profile(source)
    assert measure.occupied_count == profile.occupied_count == "1"
    assert profile.occupied_residues == ("1",)


def _powerset(values: range) -> tuple[tuple[int, ...], ...]:
    return tuple(
        combination
        for size in range(len(values) + 1)
        for combination in itertools.combinations(values, size)
    )


def _brute_residues(payload: dict[str, object]) -> tuple[int, ...]:
    request = PeriodicCongruenceUnionRequest.model_validate(payload)
    source = normalize_periodic_source(request)
    moduli = tuple(int(subset.modulus) for subset in source.subsets)
    period = math.lcm(*moduli) if moduli else 1
    subsets = tuple(
        (int(subset.modulus), {int(residue) for residue in subset.residues})
        for subset in source.subsets
    )
    return tuple(
        residue
        for residue in range(period)
        if (
            any(residue % modulus in residues for modulus, residues in subsets)
            != source.complement
        )
    )


def test_normalizes_representatives_and_merges_repeated_moduli() -> None:
    result = _measure(
        {
            "subsets": [
                {"modulus": "6", "residues": ["-1", "7", "1", "-1"]},
                {"modulus": "4", "residues": []},
                {"modulus": "6", "residues": ["13", "0"]},
            ],
            "complement": False,
        }
    )

    assert result.source.model_dump(mode="json") == {
        "subsets": [
            {"modulus": "4", "residues": []},
            {"modulus": "6", "residues": ["0", "1", "5"]},
        ],
        "complement": False,
    }
    assert result.common_period == "12"
    assert result.occupied_count == "6"
    assert result.density.as_fraction() == Fraction(1, 2)


def test_common_period_pullback_fixture_preserves_overlap_exactly() -> None:
    # This is the small finite analogue of Erdos486's
    # finiteBiasedCoveredResidues: lift each footprint to one common period,
    # take the union, and count the actual union rather than summing sizes.
    result = _profile(
        {
            "subsets": [
                {"modulus": "4", "residues": ["1"]},
                {"modulus": "6", "residues": ["1"]},
            ],
            "complement": False,
        }
    )

    assert result.common_period == "12"
    assert result.occupied_residues == ("1", "5", "7", "9")
    assert result.occupied_count == "4"
    assert result.density.as_fraction() == Fraction(1, 3)


@pytest.mark.parametrize(
    ("payload", "period", "residues"),
    [
        ({"subsets": [], "complement": False}, "1", ()),
        ({"subsets": [], "complement": True}, "1", ("0",)),
        (
            {"subsets": [{"modulus": "7", "residues": []}], "complement": False},
            "7",
            (),
        ),
        (
            {"subsets": [{"modulus": "7", "residues": []}], "complement": True},
            "7",
            tuple(str(value) for value in range(7)),
        ),
    ],
)
def test_empty_and_complement_degeneracies(
    payload: dict[str, object], period: str, residues: tuple[str, ...]
) -> None:
    result = _profile(payload)

    assert result.common_period == period
    assert result.occupied_residues == residues
    assert int(result.occupied_count) == len(residues)
    assert result.density.as_fraction() == Fraction(len(residues), int(period))


def test_noncoprime_overlap_is_not_double_counted() -> None:
    result = _profile(
        {
            "subsets": [
                {"modulus": "4", "residues": ["0"]},
                {"modulus": "6", "residues": ["2"]},
            ],
            "complement": False,
        }
    )

    assert result.occupied_residues == ("0", "2", "4", "8")
    assert result.occupied_count == "4"
    assert result.density.as_fraction() == Fraction(1, 3)


def test_measure_source_serializes_unchanged_into_profile_consumer() -> None:
    raw_payload = {
        "subsets": [
            {"modulus": "6", "residues": ["7", "-1"]},
            {"modulus": "4", "residues": ["1"]},
        ],
        "complement": False,
    }
    measure = _measure(raw_payload)

    consumer_payload = measure.source.model_dump(mode="json")
    profile = compute_periodic_congruence_union_profile(
        PeriodicCongruenceUnionProfileRequest.model_validate(consumer_payload)
    )

    assert profile.source.model_dump(mode="json") == consumer_payload
    assert profile.common_period == measure.common_period
    assert profile.occupied_count == measure.occupied_count
    assert profile.density == measure.density


def test_merged_source_with_many_residues_remains_an_unchanged_consumer_value() -> None:
    raw_payload = {
        "subsets": [
            {
                "modulus": "307",
                "residues": [str(value) for value in range(200)],
            },
            {
                "modulus": "307",
                "residues": [str(value) for value in range(200, 300)],
            },
        ],
        "complement": False,
    }
    measure = _measure(raw_payload)
    consumer_payload = measure.source.model_dump(mode="json")

    assert len(consumer_payload["subsets"][0]["residues"]) == 300
    profile = compute_periodic_congruence_union_profile(
        PeriodicCongruenceUnionProfileRequest.model_validate(consumer_payload)
    )
    assert profile.source.model_dump(mode="json") == consumer_payload
    assert profile.occupied_count == measure.occupied_count == "300"


def test_union_and_complement_partition_the_common_period() -> None:
    payload = {
        "subsets": [
            {"modulus": "4", "residues": ["0", "1"]},
            {"modulus": "6", "residues": ["1", "5"]},
        ],
        "complement": False,
    }
    union = _profile(payload)
    complement_payload = copy.deepcopy(payload)
    complement_payload["complement"] = True
    complement = _profile(complement_payload)

    assert set(union.occupied_residues).isdisjoint(complement.occupied_residues)
    assert len(union.occupied_residues) + len(complement.occupied_residues) == 12
    assert union.density.as_fraction() + complement.density.as_fraction() == 1


def test_compressed_generalized_crt_measure_avoids_period_materialization() -> None:
    result = _measure(
        {
            "subsets": [
                {"modulus": "1000003", "residues": ["0", "1"]},
                {"modulus": "1000033", "residues": ["0"]},
            ],
            "complement": False,
        }
    )

    assert result.common_period == "1000036000099"
    assert result.occupied_count == "3000067"
    assert result.density.as_fraction() == Fraction(3_000_067, 1_000_036_000_099)


def test_compressed_measure_and_sparse_materialization_agree() -> None:
    scale = 1_000_003
    payload = {
        "subsets": [
            {"modulus": str(2 * scale), "residues": ["0"]},
            {"modulus": str(3 * scale), "residues": ["0", "1"]},
        ],
        "complement": False,
    }

    measure = _measure(payload)
    profile = _profile(payload)

    assert profile.common_period == str(6 * scale)
    assert profile.occupied_residues == (
        "0",
        "1",
        str(2 * scale),
        str(3 * scale),
        str(3 * scale + 1),
        str(4 * scale),
    )
    assert measure.occupied_count == profile.occupied_count == "6"
    assert measure.density == profile.density


def test_sparse_lift_admits_large_period_with_small_exact_union() -> None:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    period = math.prod(primes)
    payload = {
        "subsets": [
            {"modulus": str(period // prime), "residues": ["0"]} for prime in primes
        ],
        "complement": False,
    }
    expected = tuple(
        sorted(
            {
                residue
                for prime in primes
                for residue in range(0, period, period // prime)
            }
        )
    )

    measure = _measure(payload)
    profile = _profile(payload)

    assert sum(primes) == 440
    assert len(expected) == 424
    assert measure.common_period == profile.common_period == str(period)
    assert measure.occupied_count == profile.occupied_count == "424"
    assert profile.occupied_residues == tuple(map(str, expected))
    assert measure.density == profile.density


def test_sparse_lift_admits_many_classes_in_one_large_modulus() -> None:
    modulus = "9" * MAX_PERIODIC_INTEGER_DIGITS
    residues = [str(value) for value in range(448)]
    payload = {
        "subsets": [{"modulus": modulus, "residues": residues}],
        "complement": False,
    }

    measure = _measure(payload)
    profile = _profile(payload)

    assert measure.common_period == profile.common_period == modulus
    assert measure.occupied_count == profile.occupied_count == "448"
    assert profile.occupied_residues == tuple(residues)
    assert measure.density.as_fraction() == Fraction(448, int(modulus))


def test_sparse_lift_work_boundary_is_exact() -> None:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    base_period = math.prod(primes)
    prime_sum = sum(primes)
    admitted_scale = MAX_SPARSE_LIFTED_ROWS // prime_sum
    rejected_scale = admitted_scale + 1

    def payload(scale: int) -> dict[str, object]:
        period = base_period * scale
        return {
            "subsets": [
                {
                    "modulus": str(base_period // prime),
                    "residues": ["0"],
                }
                for prime in primes
            ]
            + [{"modulus": str(period), "residues": []}],
            "complement": False,
        }

    admitted = _profile(payload(admitted_scale))
    assert prime_sum * admitted_scale <= MAX_SPARSE_LIFTED_ROWS
    assert len(admitted.occupied_residues) == 424 * admitted_scale

    with pytest.raises(
        OperationDomainValidationError, match="exceeds all exact execution regimes"
    ):
        compute_periodic_congruence_union_measure(
            PeriodicCongruenceUnionRequest.model_validate(payload(rejected_scale))
        )


def test_small_profiles_match_exhaustive_membership_replay() -> None:
    for residues_mod_2 in _powerset(range(2)):
        for residues_mod_3 in _powerset(range(3)):
            for complement in (False, True):
                payload: dict[str, object] = {
                    "subsets": [
                        {
                            "modulus": "2",
                            "residues": [str(value) for value in residues_mod_2],
                        },
                        {
                            "modulus": "3",
                            "residues": [str(value) for value in residues_mod_3],
                        },
                    ],
                    "complement": complement,
                }
                result = _profile(payload)
                expected = _brute_residues(payload)

                assert tuple(map(int, result.occupied_residues)) == expected
                assert int(result.occupied_count) == len(expected)
                assert result.density.as_fraction() == Fraction(len(expected), 6)


def test_measure_result_is_structural() -> None:
    result = _measure(
        {
            "subsets": [{"modulus": "5", "residues": ["0", "2"]}],
            "complement": False,
        }
    )
    serialized = result.model_dump(mode="json")

    bad_count = copy.deepcopy(serialized)
    bad_count["occupied_count"] = "3"
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionMeasureResult.model_validate(bad_count)

    bad_density = copy.deepcopy(serialized)
    bad_density["density"] = {"num": "1", "den": "5"}
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionMeasureResult.model_validate(bad_density)


def test_profile_result_is_structural() -> None:
    result = _profile(
        {
            "subsets": [{"modulus": "5", "residues": ["0", "2"]}],
            "complement": False,
        }
    )
    serialized = result.model_dump(mode="json")

    bad_period = copy.deepcopy(serialized)
    bad_period["common_period"] = "6"
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionProfileResult.model_validate(bad_period)

    bad_count = copy.deepcopy(serialized)
    bad_count["occupied_count"] = "3"
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionProfileResult.model_validate(bad_count)

    bad_density = copy.deepcopy(serialized)
    bad_density["density"] = {"num": "1", "den": "5"}
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionProfileResult.model_validate(bad_density)


def test_request_schemas_publish_aggregate_execution_and_profile_bounds() -> None:
    measure_schema = PeriodicCongruenceUnionRequest.model_json_schema()
    profile_schema = PeriodicCongruenceUnionProfileRequest.model_json_schema()

    for schema in (measure_schema, profile_schema):
        description = schema["description"]
        subsets_description = schema["properties"]["subsets"]["description"]
        assert f"{MAX_PERIODIC_SOURCE_ROWS:,} raw" in description
        assert f"{MAX_PERIODIC_SOURCE_ROWS:,} normalized" in description
        assert f"{MAX_PERIODIC_INTEGER_DIGITS} decimal digits" in description
        assert f"{MAX_PERIODIC_SOURCE_ROWS:,} raw" in subsets_description
        assert f"{MAX_PERIODIC_SOURCE_ROWS:,} normalized" in subsets_description
        assert schema["aggregate_raw_residue_row_limit"] == MAX_PERIODIC_SOURCE_ROWS
        assert (
            schema["aggregate_normalized_residue_row_limit"] == MAX_PERIODIC_SOURCE_ROWS
        )
        assert schema["common_period_digit_limit"] == MAX_PERIODIC_INTEGER_DIGITS
        assert schema["execution_passes_per_call"] == PERIODIC_EXECUTION_PASSES_PER_CALL
        assert schema["execution_regime_limits"] == {
            "period_lift": {
                "max_common_period": MAX_PERIOD_SCAN,
                "max_period_plus_lifted_rows_per_pass": MAX_PERIOD_LIFT_WORK,
                "max_period_plus_lifted_rows_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_PERIOD_LIFT_WORK
                ),
            },
            "sparse_lift": {
                "max_lifted_rows_per_pass": MAX_SPARSE_LIFTED_ROWS,
                "max_lifted_rows_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_SPARSE_LIFTED_ROWS
                ),
                "max_retained_states_per_pass": MAX_SPARSE_LIFTED_ROWS,
            },
            "inclusion_exclusion": {
                "max_retained_states_per_pass": MAX_INTERSECTION_STATES,
                "max_merges_per_pass": MAX_INTERSECTION_MERGES,
                "max_merges_per_call": (
                    PERIODIC_EXECUTION_PASSES_PER_CALL * MAX_INTERSECTION_MERGES
                ),
            },
        }

    assert (
        profile_schema["profile_materialized_residue_limit"]
        == MAX_MATERIALIZED_RESIDUES
    )
    assert (
        profile_schema["profile_full_union_period_limit"] == MAX_MATERIALIZED_RESIDUES
    )
    assert (
        profile_schema["profile_general_noncomplement_lifted_row_limit"]
        == MAX_MATERIALIZED_RESIDUES
    )
    assert (
        profile_schema["profile_nontrivial_complement_period_limit"]
        == MAX_MATERIALIZED_RESIDUES
    )
    assert profile_schema["profile_materialization_work_limit"] == (
        MAX_PERIOD_LIFT_WORK
    )
    assert profile_schema["profile_result_envelope_bytes"] == (
        PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES
    )
    assert profile_schema["profile_result_byte_limit"] == MAX_PERIODIC_RESULT_BYTES


def test_measure_accepts_exact_integer_digit_boundary() -> None:
    modulus = "9" * MAX_PERIODIC_INTEGER_DIGITS
    result = _measure(
        {
            "subsets": [{"modulus": modulus, "residues": ["0"]}],
            "complement": False,
        }
    )

    assert result.common_period == modulus
    assert result.occupied_count == "1"
    assert result.density.as_fraction() == Fraction(1, int(modulus))


def test_rejects_integer_and_lcm_above_exact_result_digit_bound() -> None:
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest.model_validate(
            {
                "subsets": [
                    {"modulus": "9" * (MAX_PERIODIC_INTEGER_DIGITS + 1), "residues": []}
                ]
            }
        )

    with pytest.raises(OperationDomainValidationError, match="common period exceeds"):
        compute_periodic_congruence_union_measure(
            PeriodicCongruenceUnionRequest.model_validate(
                {
                    "subsets": [
                        {"modulus": "1" + "0" * 199, "residues": []},
                        {"modulus": "3" * 200, "residues": []},
                    ]
                }
            )
        )


def test_source_row_and_family_boundaries_are_exact() -> None:
    full_residue_set = [str(value) for value in range(MAX_PERIODIC_SOURCE_ROWS)]
    boundary = _measure(
        {
            "subsets": [
                {
                    "modulus": str(MAX_PERIODIC_SOURCE_ROWS),
                    "residues": full_residue_set,
                }
            ]
        }
    )
    assert boundary.occupied_count == str(MAX_PERIODIC_SOURCE_ROWS)

    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest.model_validate(
            {
                "subsets": [
                    {
                        "modulus": str(MAX_PERIODIC_SOURCE_ROWS),
                        "residues": full_residue_set,
                    },
                    {"modulus": "1", "residues": ["0"]},
                ]
            }
        )

    family_boundary = [
        {"modulus": "1", "residues": []} for _ in range(MAX_PERIODIC_FAMILY_SIZE)
    ]
    PeriodicCongruenceUnionRequest.model_validate({"subsets": family_boundary})
    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest.model_validate(
            {"subsets": [*family_boundary, {"modulus": "1", "residues": []}]}
        )


def test_many_individually_valid_rows_are_admitted_to_the_aggregate_bound() -> None:
    payload = {
        "subsets": [
            {
                "modulus": "97",
                "residues": [str(value % 97) for value in range(64)],
            }
            for _ in range(MAX_PERIODIC_SOURCE_ROWS // 64)
        ],
        "complement": False,
    }

    result = _measure(payload)

    assert len(cast(list[object], payload["subsets"])) == MAX_PERIODIC_FAMILY_SIZE
    assert result.occupied_count == "64"
    assert result.density.as_fraction() == Fraction(64, 97)


def test_aggregate_row_bound_rejects_before_constructing_row_models() -> None:
    payload = {
        "subsets": [
            {
                "modulus": "2",
                "residues": [None] * MAX_PERIODIC_SOURCE_ROWS,
            }
            for _ in range(MAX_PERIODIC_FAMILY_SIZE)
        ],
        "complement": False,
    }

    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest.model_validate(payload)


def test_oversized_family_is_rejected_without_normalizing_entries() -> None:
    class UnnormalizableEntry(Mapping[str, object]):
        """A row whose normalization would fail if it were ever copied."""

        def __init__(self) -> None:
            self._data: dict[str, object] = {"modulus": "2", "residues": ["0"]}

        def __getitem__(self, key: str) -> object:
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("entry normalization ran past the 64-subset bound")

        def __len__(self) -> int:
            return len(self._data)

    payload = {
        "subsets": [UnnormalizableEntry() for _ in range(MAX_PERIODIC_FAMILY_SIZE + 1)],
        "complement": False,
    }

    with expect_validation("number_theory."):
        PeriodicCongruenceUnionProfileRequest.model_validate(payload)


def test_constructor_path_rejects_oversized_families_and_raw_rows() -> None:
    boundary_rows = tuple(
        PeriodicCongruenceSubsetInput(
            modulus="97",
            residues=tuple(str(value % 97) for value in range(64)),
        )
        for _ in range(MAX_PERIODIC_FAMILY_SIZE)
    )
    accepted = PeriodicCongruenceUnionRequest(subsets=boundary_rows, complement=False)
    assert len(normalize_periodic_source(accepted).subsets) == 1

    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest(
            subsets=(
                *boundary_rows,
                PeriodicCongruenceSubsetInput(modulus="97", residues=()),
            ),
            complement=False,
        )

    with expect_validation("number_theory."):
        PeriodicCongruenceUnionRequest(
            subsets=(
                PeriodicCongruenceSubsetInput(
                    modulus="2",
                    residues=("0",) * MAX_PERIODIC_SOURCE_ROWS,
                ),
                PeriodicCongruenceSubsetInput(modulus="2", residues=("1",)),
            ),
            complement=False,
        )


def test_materialized_period_boundary_is_separate_from_measure() -> None:
    large_sparse = {
        "subsets": [
            {
                "modulus": "9" * MAX_PERIODIC_INTEGER_DIGITS,
                "residues": ["0"],
            }
        ],
        "complement": False,
    }
    result = _profile(large_sparse)
    assert result.occupied_residues == ("0",)

    complemented_boundary = {
        "subsets": [{"modulus": str(MAX_MATERIALIZED_RESIDUES), "residues": []}],
        "complement": True,
    }
    boundary_profile = _profile(complemented_boundary)
    assert len(boundary_profile.occupied_residues) == MAX_MATERIALIZED_RESIDUES
    assert boundary_profile.occupied_residues[-1] == str(MAX_MATERIALIZED_RESIDUES - 1)

    measure_only = {
        "subsets": [
            {
                "modulus": str(MAX_MATERIALIZED_RESIDUES + 1),
                "residues": [],
            }
        ],
        "complement": True,
    }
    assert _measure(measure_only).occupied_count == str(MAX_MATERIALIZED_RESIDUES + 1)
    with pytest.raises(OperationDomainValidationError, match="common period"):
        _profile(measure_only)


def test_full_subset_shortcuts_count_and_complement_materialization() -> None:
    huge_period = "9" * MAX_PERIODIC_INTEGER_DIGITS
    source = [
        {"modulus": "1", "residues": ["0"]},
        {"modulus": huge_period, "residues": []},
    ]

    full_measure = _measure({"subsets": source, "complement": False})
    empty_complement = _profile({"subsets": source, "complement": True})

    assert full_measure.common_period == huge_period
    assert full_measure.occupied_count == huge_period
    assert full_measure.density.as_fraction() == 1
    assert empty_complement.occupied_residues == ()
    assert empty_complement.occupied_count == "0"
    assert empty_complement.density.as_fraction() == 0

    with pytest.raises(OperationDomainValidationError, match="materialized full union"):
        _profile({"subsets": source, "complement": False})


def test_materialized_union_row_bound_is_checked_before_lifting() -> None:
    boundary_payload = {
        "subsets": [
            {"modulus": "2", "residues": ["0"]},
            {"modulus": str(2 * MAX_MATERIALIZED_RESIDUES), "residues": []},
        ],
        "complement": False,
    }
    boundary_profile = _profile(boundary_payload)
    assert int(boundary_profile.occupied_count) == MAX_MATERIALIZED_RESIDUES

    rejected_payload = {
        "subsets": [
            {"modulus": "2", "residues": ["0"]},
            {
                "modulus": str(2 * (MAX_MATERIALIZED_RESIDUES + 1)),
                "residues": [],
            },
        ],
        "complement": False,
    }
    with pytest.raises(OperationDomainValidationError, match="materialized union"):
        _profile(rejected_payload)


def test_profile_accounts_for_retained_source_and_wide_residue_output_bytes() -> None:
    base_modulus = 10**250
    output_rows = 50_000
    payload = {
        "subsets": [
            {"modulus": str(base_modulus), "residues": ["0"]},
            {"modulus": str(base_modulus * output_rows), "residues": []},
        ],
        "complement": False,
    }

    assert _measure(payload).occupied_count == str(output_rows)
    with pytest.raises(OperationDomainValidationError, match="canonical output budget"):
        _profile(payload)


def test_compressed_intersection_work_boundary_rejects_before_backend() -> None:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    boundary_payload = {
        "subsets": [{"modulus": str(prime), "residues": ["0"]} for prime in primes[:-1]]
    }
    request = PeriodicCongruenceUnionRequest.model_validate(boundary_payload)
    assert (1 << len(request.subsets)) - 1 == MAX_INTERSECTION_STATES
    result = compute_periodic_congruence_union_measure(request)
    period = math.prod(primes[:-1])
    assert int(result.occupied_count) == period - math.prod(
        prime - 1 for prime in primes[:-1]
    )

    rejected_payload = copy.deepcopy(boundary_payload)
    rejected_payload["subsets"].append({"modulus": str(primes[-1]), "residues": ["0"]})
    with pytest.raises(
        OperationDomainValidationError, match="exceeds all exact execution regimes"
    ):
        compute_periodic_congruence_union_measure(
            PeriodicCongruenceUnionRequest.model_validate(rejected_payload)
        )


def test_non_coprime_generalized_crt_replays_scaled_union() -> None:
    base_payload: dict[str, object] = {
        "subsets": [
            {"modulus": "4", "residues": ["0", "1"]},
            {"modulus": "6", "residues": ["1", "2"]},
        ]
    }
    base_residues = _brute_residues(base_payload)
    assert base_residues == (0, 1, 2, 4, 5, 7, 8, 9)

    scale = MAX_PERIOD_SCAN // 12 + 1
    large_multiple = 12 * scale
    payload = {
        "subsets": [
            {"modulus": "4", "residues": ["0", "1"]},
            {"modulus": "6", "residues": ["1", "2"]},
            {"modulus": str(large_multiple), "residues": []},
        ]
    }
    request = PeriodicCongruenceUnionRequest.model_validate(payload)
    assert require_admitted_periodic_source(
        normalize_periodic_source(request)
    ).method == ("INCLUSION_EXCLUSION")

    union = compute_periodic_congruence_union_measure(request)
    assert union.common_period == str(large_multiple)
    assert union.occupied_count == str(len(base_residues) * scale)

    complement = _measure({**payload, "complement": True})
    assert complement.common_period == str(large_multiple)
    assert complement.occupied_count == str((12 - len(base_residues)) * scale)
