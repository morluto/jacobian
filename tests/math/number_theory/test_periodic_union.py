from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory._periodic_union import (
    MAX_PERIODIC_COMMON_PERIOD,
    MAX_PERIODIC_FAMILY_SIZE,
    MAX_PERIODIC_MATERIALIZED_RESIDUES,
    MAX_PERIODIC_SOURCE_RESIDUES,
    PeriodicResidueSubset,
    PeriodicUnionProfileRequest,
    PeriodicUnionProfileResult,
    compute_periodic_union_profile,
)
from jacobian.math.number_theory._tools import TOOLS


def _request(
    *subsets: PeriodicResidueSubset,
    complement: bool = False,
    result_mode: str = "materialize_residues",
) -> PeriodicUnionProfileRequest:
    return PeriodicUnionProfileRequest.model_validate(
        {
            "subsets": [subset.model_dump(mode="json") for subset in subsets],
            "complement": complement,
            "result_mode": result_mode,
        }
    )


def _occupied_subset(result: PeriodicUnionProfileResult) -> PeriodicResidueSubset:
    assert result.occupied_subset is not None
    return result.occupied_subset


def test_periodic_union_operation_is_public_and_example_is_exact() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "congruence.periodic_union.profile.compute"
    )

    result = operation.run(
        operation.request_type.model_validate(operation.examples[0].input)
    )

    assert result.common_period == 30
    assert result.occupied_count == 9
    assert result.density == CanonicalRational(num="3", den="10")
    assert _occupied_subset(result) == PeriodicResidueSubset(
        modulus=30,
        residues=(1, 4, 7, 9, 13, 14, 19, 24, 25),
    )


def test_advertised_example_parses_through_strict_json() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "congruence.periodic_union.profile.compute"
    )

    request = PeriodicUnionProfileRequest.model_validate_json(
        json.dumps(operation.examples[0].input),
        strict=True,
    )

    assert request.subsets[0] == PeriodicResidueSubset(modulus=6, residues=(1,))


def test_empty_family_has_period_one_with_union_and_complement_conventions() -> None:
    union = compute_periodic_union_profile(_request())
    complement = compute_periodic_union_profile(_request(complement=True))

    assert (union.common_period, union.occupied_count, _occupied_subset(union)) == (
        1,
        0,
        PeriodicResidueSubset(modulus=1, residues=()),
    )
    assert union.density == CanonicalRational(num="0", den="1")
    assert (
        complement.common_period,
        complement.occupied_count,
        _occupied_subset(complement),
    ) == (1, 1, PeriodicResidueSubset(modulus=1, residues=(0,)))
    assert complement.density == CanonicalRational(num="1", den="1")


def test_repeated_moduli_and_empty_subsets_preserve_declared_common_period() -> None:
    result = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(modulus=4, residues=()),
            PeriodicResidueSubset(modulus=4, residues=(0, 2)),
            PeriodicResidueSubset(modulus=4, residues=(1,)),
        )
    )

    assert result.common_period == 4
    assert _occupied_subset(result) == PeriodicResidueSubset(
        modulus=4, residues=(0, 1, 2)
    )
    assert result.occupied_count == 3
    assert result.density == CanonicalRational(num="3", den="4")


def test_count_only_and_materialized_modes_agree_when_both_are_admitted() -> None:
    subsets = (
        PeriodicResidueSubset(modulus=6, residues=(0, 5)),
        PeriodicResidueSubset(modulus=10, residues=(3,)),
    )
    count_only = compute_periodic_union_profile(
        _request(*subsets, result_mode="count_only")
    )
    materialized = compute_periodic_union_profile(_request(*subsets))

    assert count_only.common_period == materialized.common_period
    assert count_only.occupied_count == materialized.occupied_count
    assert count_only.density == materialized.density
    assert count_only.occupied_subset is None
    assert len(_occupied_subset(materialized).residues) == count_only.occupied_count


def test_union_and_complement_are_exact_duals() -> None:
    subsets = (
        PeriodicResidueSubset(modulus=4, residues=(0,)),
        PeriodicResidueSubset(modulus=6, residues=(1, 5)),
    )
    union = compute_periodic_union_profile(_request(*subsets))
    complement = compute_periodic_union_profile(_request(*subsets, complement=True))

    assert union.common_period == complement.common_period == 12
    assert union.occupied_count + complement.occupied_count == 12
    assert set(_occupied_subset(union).residues).isdisjoint(
        _occupied_subset(complement).residues
    )
    assert set(_occupied_subset(union).residues) | set(
        _occupied_subset(complement).residues
    ) == set(range(12))
    assert union.density.as_fraction() + complement.density.as_fraction() == 1


def test_small_profiles_match_direct_membership_exhaustively() -> None:
    modulo_two = tuple(
        PeriodicResidueSubset(modulus=2, residues=residues)
        for length in range(3)
        for residues in combinations(range(2), length)
    )
    modulo_three = tuple(
        PeriodicResidueSubset(modulus=3, residues=residues)
        for length in range(4)
        for residues in combinations(range(3), length)
    )

    for left in modulo_two:
        for right in modulo_three:
            for complement in (False, True):
                request = _request(left, right, complement=complement)
                result = compute_periodic_union_profile(request)
                expected = tuple(
                    residue
                    for residue in range(6)
                    if (
                        (
                            residue % left.modulus in left.residues
                            or residue % right.modulus in right.residues
                        )
                        != complement
                    )
                )
                assert _occupied_subset(result).residues == expected
                assert result.occupied_count == len(expected)


@pytest.mark.parametrize(
    "residues",
    ((1, 0), (0, 0), (-1,), (3,)),
)
def test_residue_subset_rejects_noncanonical_rows(
    residues: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="residues"):
        PeriodicResidueSubset(modulus=3, residues=residues)


def test_request_rejects_duplicate_or_noncanonical_family_order() -> None:
    left = PeriodicResidueSubset(modulus=3, residues=(0,))
    right = PeriodicResidueSubset(modulus=5, residues=(1,))

    with pytest.raises(ValidationError, match="unique and canonically ordered"):
        _request(right, left)
    with pytest.raises(ValidationError, match="unique and canonically ordered"):
        _request(left, left)


def test_source_residue_boundary_is_derived_across_the_family() -> None:
    half = MAX_PERIODIC_MATERIALIZED_RESIDUES // 2
    even = PeriodicResidueSubset(
        modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES,
        residues=tuple(range(0, MAX_PERIODIC_MATERIALIZED_RESIDUES, 2)),
    )
    odd = PeriodicResidueSubset(
        modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES,
        residues=tuple(range(1, MAX_PERIODIC_MATERIALIZED_RESIDUES, 2)),
    )

    accepted = compute_periodic_union_profile(
        _request(even, odd, result_mode="count_only")
    )
    assert len(even.residues) == len(odd.residues) == half
    assert accepted.occupied_count == MAX_PERIODIC_MATERIALIZED_RESIDUES

    payload = {
        "subsets": [
            even.model_dump(mode="json"),
            odd.model_dump(mode="json"),
            {"modulus": MAX_PERIODIC_MATERIALIZED_RESIDUES, "residues": [0]},
        ],
        "result_mode": "count_only",
    }
    with pytest.raises(ValidationError, match="32,768-source-residue"):
        PeriodicUnionProfileRequest.model_validate(payload)


def test_common_period_boundary_is_checked_before_materialization() -> None:
    accepted = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(modulus=MAX_PERIODIC_COMMON_PERIOD, residues=()),
            result_mode="count_only",
        )
    )
    assert accepted.common_period == MAX_PERIODIC_COMMON_PERIOD

    with pytest.raises(ValidationError, match="least common period"):
        _request(
            PeriodicResidueSubset(modulus=2, residues=()),
            PeriodicResidueSubset(modulus=999_983, residues=()),
            result_mode="count_only",
        )


def test_mark_write_boundary_uses_aggregated_lifted_cardinality_per_pass() -> None:
    full_moduli = (1, 2, 4, 5, 8, 10, 16, 20)
    at_limit = (
        *(
            PeriodicResidueSubset(modulus=modulus, residues=tuple(range(modulus)))
            for modulus in full_moduli
        ),
        PeriodicResidueSubset(modulus=1_000_000, residues=()),
    )

    result = compute_periodic_union_profile(
        _request(*at_limit, result_mode="count_only")
    )
    assert result.occupied_count == 1_000_000

    above_limit = tuple(
        sorted(
            (
                *at_limit,
                PeriodicResidueSubset(modulus=25, residues=tuple(range(25))),
            ),
            key=lambda subset: (subset.modulus, subset.residues),
        )
    )
    with pytest.raises(ValidationError, match="per-pass work bound"):
        _request(*above_limit, result_mode="count_only")


def test_count_only_admits_dense_large_profile_that_output_mode_rejects() -> None:
    dense = (
        PeriodicResidueSubset(modulus=1, residues=(0,)),
        PeriodicResidueSubset(modulus=1_000_000, residues=()),
    )

    counted = compute_periodic_union_profile(_request(*dense, result_mode="count_only"))
    assert counted.occupied_count == 1_000_000
    assert counted.occupied_subset is None

    with pytest.raises(ValidationError, match="use count_only"):
        _request(*dense)


def test_output_admission_accepts_provably_sparse_large_profiles() -> None:
    sparse = PeriodicResidueSubset(modulus=1_000_000, residues=(999_999,))
    result = compute_periodic_union_profile(_request(sparse))
    assert _occupied_subset(result) == sparse

    full = (
        PeriodicResidueSubset(modulus=1, residues=(0,)),
        PeriodicResidueSubset(modulus=1_000_000, residues=()),
    )
    complement = compute_periodic_union_profile(_request(*full, complement=True))
    assert complement.occupied_count == 0
    assert _occupied_subset(complement) == PeriodicResidueSubset(
        modulus=1_000_000, residues=()
    )


def test_materialized_union_boundary_round_trips_into_the_same_operation() -> None:
    at_limit = _request(
        PeriodicResidueSubset(modulus=1, residues=(0,)),
        PeriodicResidueSubset(modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES, residues=()),
    )
    produced = compute_periodic_union_profile(at_limit)
    occupied = _occupied_subset(produced)
    assert len(occupied.residues) == MAX_PERIODIC_MATERIALIZED_RESIDUES

    serialized = occupied.model_dump(mode="json")
    consumed = compute_periodic_union_profile(
        PeriodicUnionProfileRequest.model_validate(
            {"subsets": [serialized], "result_mode": "materialize_residues"}
        )
    )
    assert _occupied_subset(consumed).model_dump(mode="json") == serialized

    above_limit = (
        PeriodicResidueSubset(modulus=1, residues=(0,)),
        PeriodicResidueSubset(
            modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES + 1, residues=()
        ),
    )
    counted = compute_periodic_union_profile(
        _request(*above_limit, result_mode="count_only")
    )
    assert counted.occupied_count == MAX_PERIODIC_MATERIALIZED_RESIDUES + 1
    assert counted.occupied_subset is None

    with pytest.raises(ValidationError, match="32,768-residue output bound"):
        _request(*above_limit)


def test_materialized_complement_boundary_is_exact() -> None:
    accepted = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(
                modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES, residues=()
            ),
            complement=True,
        )
    )
    assert len(_occupied_subset(accepted).residues) == (
        MAX_PERIODIC_MATERIALIZED_RESIDUES
    )

    above_limit = PeriodicResidueSubset(
        modulus=MAX_PERIODIC_MATERIALIZED_RESIDUES + 1, residues=()
    )
    counted = compute_periodic_union_profile(
        _request(above_limit, complement=True, result_mode="count_only")
    )
    assert counted.occupied_count == MAX_PERIODIC_MATERIALIZED_RESIDUES + 1
    assert counted.occupied_subset is None

    with pytest.raises(ValidationError, match="32,768-residue output bound"):
        _request(above_limit, complement=True)


def test_same_modulus_aggregation_admits_an_empty_large_complement() -> None:
    subsets = (
        PeriodicResidueSubset(modulus=2, residues=(0,)),
        PeriodicResidueSubset(modulus=2, residues=(1,)),
        PeriodicResidueSubset(modulus=1_000_000, residues=()),
    )

    result = compute_periodic_union_profile(_request(*subsets, complement=True))

    assert result.source.subsets == subsets
    assert result.common_period == 1_000_000
    assert result.occupied_count == 0
    assert _occupied_subset(result) == PeriodicResidueSubset(
        modulus=1_000_000, residues=()
    )


def test_empty_materialized_output_round_trips_with_its_period() -> None:
    produced = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(modulus=1, residues=(0,)),
            PeriodicResidueSubset(modulus=1_000_000, residues=()),
            complement=True,
        )
    )
    serialized = _occupied_subset(produced).model_dump(mode="json")

    consumed = compute_periodic_union_profile(
        PeriodicUnionProfileRequest.model_validate(
            {"subsets": [serialized], "result_mode": "materialize_residues"}
        )
    )

    assert consumed.common_period == 1_000_000
    assert _occupied_subset(consumed).model_dump(mode="json") == serialized


def test_family_size_boundary_allows_repeated_moduli() -> None:
    accepted_subsets = tuple(
        PeriodicResidueSubset(
            modulus=MAX_PERIODIC_FAMILY_SIZE,
            residues=(residue,),
        )
        for residue in range(MAX_PERIODIC_FAMILY_SIZE)
    )
    accepted = compute_periodic_union_profile(
        _request(*accepted_subsets, result_mode="count_only")
    )
    assert accepted.occupied_count == MAX_PERIODIC_FAMILY_SIZE

    rejected_payload = {
        "subsets": [
            {"modulus": MAX_PERIODIC_FAMILY_SIZE + 1, "residues": [residue]}
            for residue in range(MAX_PERIODIC_FAMILY_SIZE + 1)
        ],
        "result_mode": "count_only",
    }
    with pytest.raises(ValidationError, match="64-subset bound"):
        PeriodicUnionProfileRequest.model_validate(rejected_payload)

    with pytest.raises(ValidationError, match="64-subset bound"):
        PeriodicUnionProfileRequest(
            subsets=tuple(
                PeriodicResidueSubset(modulus=MAX_PERIODIC_FAMILY_SIZE, residues=())
                for _ in range(MAX_PERIODIC_FAMILY_SIZE + 1)
            )
        )


def test_oversized_family_is_rejected_without_normalizing_entries() -> None:
    class UnnormalizableEntry(Mapping[str, object]):
        """A row whose normalization would fail if it were ever copied."""

        def __init__(self) -> None:
            self._data: dict[str, object] = {"modulus": 2, "residues": [0]}

        def __getitem__(self, key: str) -> object:
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("entry normalization ran past the 64-subset bound")

        def __len__(self) -> int:
            return len(self._data)

    payload = {
        "subsets": [UnnormalizableEntry() for _ in range(MAX_PERIODIC_FAMILY_SIZE + 1)],
        "result_mode": "count_only",
    }

    with pytest.raises(ValidationError, match="family exceeds the 64-subset bound"):
        PeriodicUnionProfileRequest.model_validate(payload)


def test_oversized_row_is_rejected_before_copying_its_residues() -> None:
    class UncopyableResidues(list[int]):
        """A residue list whose copy would fail if it were ever materialized."""

        def __iter__(self) -> Iterator[int]:
            raise AssertionError(
                "oversized residue list was copied past the source-residue bound"
            )

    payload = {
        "subsets": [
            {
                "modulus": MAX_PERIODIC_SOURCE_RESIDUES,
                "residues": UncopyableResidues(range(MAX_PERIODIC_SOURCE_RESIDUES + 1)),
            }
        ],
        "result_mode": "count_only",
    }

    with pytest.raises(ValidationError, match="32,768-source-residue"):
        PeriodicUnionProfileRequest.model_validate(payload)


def test_small_erdos_486_periodic_footprint_analogue() -> None:
    """Check the source's finite footprint invariant on a deliberately small family.

    ShouqiaoW/erdos@d28713ac8245ca86a686b8c67370a8d19d81b242 defines
    endpoint-cylinder coverage in ``486/lean/Erdos486/BiasedColoring.lean`` and
    identifies its occupied-count ratio in
    ``486/lean/Erdos486/BiasedFootprintAverage.lean``. This three-cylinder
    analogue does not reproduce the real j >= 400 construction, which is outside
    this operation's admitted materialized envelope.
    """

    endpoint_moduli = ((11, 4), (14, 6), (20, 9))
    subsets = tuple(
        PeriodicResidueSubset(modulus=modulus, residues=(endpoint % modulus,))
        for endpoint, modulus in endpoint_moduli
    )
    result = compute_periodic_union_profile(_request(*subsets))
    expected = tuple(
        residue
        for residue in range(result.common_period)
        if any(
            residue % modulus == endpoint % modulus
            for endpoint, modulus in endpoint_moduli
        )
    )

    assert result.common_period == 36
    assert _occupied_subset(result).residues == expected
    assert result.occupied_count == len(expected)
    assert result.density.as_fraction() == Fraction(len(expected), result.common_period)


def test_schema_publishes_canonical_and_conditional_contracts() -> None:
    request_schema = PeriodicUnionProfileRequest.model_json_schema()
    subset_schema = request_schema["$defs"]["PeriodicResidueSubset"]
    result_schema = PeriodicUnionProfileResult.model_json_schema()
    request_description = " ".join(request_schema["description"].split())

    assert (
        "strictly increasing" in subset_schema["properties"]["residues"]["description"]
    )
    assert "[0, modulus)" in subset_schema["properties"]["residues"]["description"]
    assert (
        "ordered strictly by (modulus, residues)"
        in request_schema["properties"]["subsets"]["description"]
    )
    assert (
        "Repeated moduli are allowed"
        in request_schema["properties"]["subsets"]["description"]
    )
    assert (
        "32,768 residue rows in total"
        in request_schema["properties"]["subsets"]["description"]
    )
    assert (
        "complement inside [0, L)"
        in request_schema["properties"]["complement"]["description"]
    )
    assert (
        "32,768 output residues"
        in request_schema["properties"]["result_mode"]["description"]
    )
    assert "8,000,000" in request_description
    assert "16,000,000" in request_description
    assert "two such scans" in request_description
    assert (
        "modulus equal to common_period"
        in result_schema["properties"]["occupied_subset"]["description"]
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("common_period", 29, "common period"),
        ("occupied_count", 8, "occupied count"),
        ("density", {"num": "1", "den": "3"}, "density"),
        (
            "occupied_subset",
            {"modulus": 30, "residues": [1, 4]},
            "occupied subset",
        ),
    ),
)
def test_result_replay_rejects_forged_profiles(
    field: str,
    replacement: object,
    message: str,
) -> None:
    result = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(modulus=6, residues=(1,)),
            PeriodicResidueSubset(modulus=10, residues=(4,)),
            PeriodicResidueSubset(modulus=15, residues=(9,)),
        )
    )
    payload = result.model_dump(mode="json")
    payload[field] = replacement

    with pytest.raises(ValidationError, match=message):
        PeriodicUnionProfileResult.model_validate(payload)


def test_count_only_result_cannot_carry_a_materialized_profile() -> None:
    result = compute_periodic_union_profile(
        _request(
            PeriodicResidueSubset(modulus=5, residues=(1,)),
            result_mode="count_only",
        )
    )
    payload = result.model_dump(mode="json")
    payload["occupied_subset"] = {"modulus": 5, "residues": [1]}

    with pytest.raises(ValidationError, match="occupied subset"):
        PeriodicUnionProfileResult.model_validate(payload)


@pytest.mark.parametrize(
    ("source_mutation", "message"),
    (
        ({"subsets": [{"modulus": 5, "residues": [2]}]}, "occupied subset"),
        ({"complement": True}, "occupied count"),
        ({"result_mode": "count_only"}, "occupied subset"),
    ),
)
def test_result_replay_rejects_source_complement_and_mode_mutations(
    source_mutation: dict[str, object],
    message: str,
) -> None:
    result = compute_periodic_union_profile(
        _request(PeriodicResidueSubset(modulus=5, residues=(1,)))
    )
    payload = result.model_dump(mode="json")
    payload["source"].update(source_mutation)

    with pytest.raises(ValidationError, match=message):
        PeriodicUnionProfileResult.model_validate(payload)
