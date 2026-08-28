"""Behavioral tests for finite delta-matroid recognition."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.greedoids import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids import delta as delta_matroids
from jacobian.math.combinatorics.matroids.delta import FiniteDeltaMatroid
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.combinatorics.matroids.delta._operations import (
    compute_from_feasible_sets,
)
from jacobian.math.combinatorics.matroids.delta._tools import TOOLS


def _two_element_delta_matroid(*, scrambled: bool = False) -> FiniteFeasibleSetSystem:
    feasible = ((0,), (), (0, 1), (1,)) if scrambled else ((), (0,), (0, 1), (1,))
    return FiniteFeasibleSetSystem(ground=("a", "b"), feasible=feasible)


def test_catalog_contains_only_audited_agent_outcome() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "delta_matroid.from_feasible_sets.compute",
    }


def test_complete_feasible_family_constructs_canonical_delta_matroid() -> None:
    result = compute_from_feasible_sets(
        DeltaMatroidFromFeasibleSetsRequest(system=_two_element_delta_matroid())
    )

    assert result.status == "DELTA_MATROID"
    assert result.obstruction is None
    assert result.delta_matroid is not None
    assert result.delta_matroid.ground == ("a", "b")
    assert result.delta_matroid.feasible == ((), (0,), (0, 1), (1,))


def test_empty_ground_identity_delta_matroid_has_native_and_wire_replay() -> None:
    system = FiniteFeasibleSetSystem(ground=(), feasible=((),))

    native_result = delta_matroids.from_feasible_sets(system)
    assert native_result.status == "DELTA_MATROID"
    assert native_result.delta_matroid == FiniteDeltaMatroid(ground=(), feasible=((),))

    wire_result = compute_from_feasible_sets(
        DeltaMatroidFromFeasibleSetsRequest(system=system)
    )
    assert wire_result == native_result
    assert (
        DeltaMatroidRecognitionResult.model_validate_json(wire_result.model_dump_json())
        == wire_result
    )


def test_row_order_is_not_mathematical_but_output_is_canonical() -> None:
    result = delta_matroids.from_feasible_sets(
        _two_element_delta_matroid(scrambled=True)
    )

    assert result.status == "DELTA_MATROID"
    assert result.delta_matroid is not None
    assert result.delta_matroid.feasible == ((), (0,), (0, 1), (1,))


def test_empty_complete_family_has_typed_first_obstruction() -> None:
    result = delta_matroids.from_feasible_sets(
        FiniteFeasibleSetSystem(ground=("a",), feasible=())
    )

    assert result.status == "NOT_A_DELTA_MATROID"
    assert result.delta_matroid is None
    assert result.obstruction is not None
    assert result.obstruction.kind == "EMPTY_FEASIBLE_FAMILY"


def test_symmetric_exchange_failure_is_deterministic_and_exhaustive() -> None:
    result = delta_matroids.from_feasible_sets(
        FiniteFeasibleSetSystem(
            ground=("a", "b", "c"),
            feasible=((), (0, 1), (2,)),
        )
    )

    assert result.status == "NOT_A_DELTA_MATROID"
    assert result.obstruction is not None
    assert result.obstruction.kind == "SYMMETRIC_EXCHANGE"
    assert result.obstruction.left_feasible == (0, 1)
    assert result.obstruction.right_feasible == (2,)
    assert result.obstruction.element == 2
    assert result.obstruction.symmetric_difference == (0, 1, 2)


def test_forged_valid_result_is_structurally_parseable() -> None:
    result = delta_matroids.from_feasible_sets(_two_element_delta_matroid())
    forged = result.model_dump(mode="json")
    assert forged["delta_matroid"] is not None
    forged["delta_matroid"]["ground"][0] = "changed"

    assert DeltaMatroidRecognitionResult.model_validate(forged)


def test_result_round_trips_without_replaying_its_retained_source() -> None:
    result = delta_matroids.from_feasible_sets(_two_element_delta_matroid())
    assert (
        DeltaMatroidRecognitionResult.model_validate_json(result.model_dump_json())
        == result
    )

    forged = result.model_dump(mode="json")
    forged["source"]["ground"][0] = "changed"
    assert DeltaMatroidRecognitionResult.model_validate(forged)


def test_invalid_delta_matroid_value_is_rejected_at_its_value_boundary() -> None:
    feasible = ((), (0, 1), (2,))
    with pytest.raises(ValidationError) as error:
        FiniteDeltaMatroid(
            ground=("a", "b", "c"),
            feasible=feasible,
        )
    assert error.value.errors()[0]["type"] == "delta_matroid.exchange_axiom_failed"


def test_sparse_family_beyond_any_fixed_ground_cap_is_recognized() -> None:
    # Sixty-five short labels with only the empty feasible set carry zero
    # memberships and zero exchange candidates, so every derived budget admits
    # them; no fixed ground-size ceiling may exclude this valid delta-matroid.
    system = FiniteFeasibleSetSystem(
        ground=tuple(f"e{index}" for index in range(65)),
        feasible=((),),
    )

    native_result = delta_matroids.from_feasible_sets(system)
    assert native_result.status == "DELTA_MATROID"
    assert native_result.delta_matroid == FiniteDeltaMatroid(
        ground=system.ground,
        feasible=((),),
    )

    wire_result = compute_from_feasible_sets(
        DeltaMatroidFromFeasibleSetsRequest(system=system)
    )
    assert wire_result == native_result
    assert (
        DeltaMatroidRecognitionResult.model_validate_json(wire_result.model_dump_json())
        == wire_result
    )


def test_label_byte_budget_bounds_ground_count_without_a_fixed_cap() -> None:
    # Ground labels must be unique, so the UTF-8 label-byte envelope itself
    # bounds the element count: exactly 1,024 distinct two-byte labels fit,
    # while a 1,025th exceeds the byte budget and names that quantity.
    def _labels(count: int) -> tuple[str, ...]:
        return tuple(
            chr(33 + index // 32) + chr(33 + index % 32) for index in range(count)
        )

    admitted = FiniteFeasibleSetSystem(ground=_labels(1_024), feasible=((),))
    result = delta_matroids.from_feasible_sets(admitted)
    assert result.status == "DELTA_MATROID"
    assert result.delta_matroid is not None
    assert len(result.delta_matroid.ground) == 1_024

    oversized = FiniteFeasibleSetSystem(ground=_labels(1_025), feasible=((),))
    request = DeltaMatroidFromFeasibleSetsRequest(system=oversized)
    with pytest.raises(ValueError, match="ground labels exceed"):
        compute_from_feasible_sets(request)


def test_non_utf8_representable_ground_labels_are_rejected_not_host_errors() -> None:
    # An unpaired surrogate is structurally well formed for the shared carrier
    # but has no UTF-8 byte length, so admission must reject it with a
    # controlled validation error instead of leaking UnicodeEncodeError.
    system = FiniteFeasibleSetSystem(ground=("\ud800",), feasible=((),))

    with pytest.raises(ValueError, match="UTF-8-representable"):
        delta_matroids.from_feasible_sets(system)

    request = DeltaMatroidFromFeasibleSetsRequest(system=system)
    with pytest.raises(ValueError, match="UTF-8-representable"):
        compute_from_feasible_sets(request)

    with pytest.raises(ValidationError) as error:
        FiniteDeltaMatroid(ground=("\ud800",), feasible=((),))
    assert error.value.errors()[0]["type"] == "delta_matroid.labels_not_utf8"


def test_request_schema_exposes_every_delta_specific_admission_limit() -> None:
    schema = DeltaMatroidFromFeasibleSetsRequest.model_json_schema()

    assert schema["admission_limits"] == {
        "max_feasible_set_memberships": 1_024,
        "max_ground_label_utf8_bytes": 2_048,
        "max_symmetric_exchange_candidate_checks_per_replay": 250_000,
        "max_result_bytes": 65_536,
    }
    assert "no separate ground-size or row-count caps" in schema["description"]
    assert "structural only" in schema["description"]


def test_short_row_family_beyond_any_row_cap_is_recognized() -> None:
    # Every subset of a sixteen-element ground with size at most two: 137 rows
    # and 220,832 ordered exchange candidate checks. The family fits every
    # derived membership, candidate-work, and result bound, so no row-count
    # ceiling may exclude it.
    feasible: list[tuple[int, ...]] = [()]
    feasible.extend((index,) for index in range(16))
    feasible.extend(
        (left, right) for left in range(15) for right in range(left + 1, 16)
    )
    system = FiniteFeasibleSetSystem(
        ground=tuple(f"e{index}" for index in range(16)),
        feasible=tuple(feasible),
    )

    result = delta_matroids.from_feasible_sets(system)

    assert result.status == "DELTA_MATROID"
    assert result.delta_matroid == FiniteDeltaMatroid(
        ground=system.ground,
        feasible=tuple(sorted(feasible)),
    )


def test_membership_envelope_rejects_wide_families_without_a_row_cap() -> None:
    # Six hundred distinct pairs carry 1,200 memberships, past the membership
    # envelope; rejection names the controlling quantity rather than a row cap.
    feasible = []
    for index in range(25):
        for offset in range(1, 25):
            feasible.append((index, index + offset))
    request = DeltaMatroidFromFeasibleSetsRequest(
        system=FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(50)),
            feasible=tuple(feasible),
        )
    )
    with pytest.raises(ValueError, match="memberships exceed"):
        compute_from_feasible_sets(request)


def test_native_admission_rejects_exchange_candidate_space_before_axiom_pass() -> None:
    # The 128 even-parity subsets of an eight-element ground set are a compact
    # input whose complete ordered exchange candidate space exceeds the public
    # budget. Recognition must reject the work before attempting the axiom.
    # Add the parity bit so every seven-bit word gives one distinct even subset.
    feasible = tuple(
        row if len(row) % 2 == 0 else (*row, 7)
        for row in (
            tuple(bit for bit in range(7) if (index >> bit) & 1) for index in range(128)
        )
    )
    request = DeltaMatroidFromFeasibleSetsRequest(
        system=FiniteFeasibleSetSystem(
            ground=tuple(f"e{index}" for index in range(8)),
            feasible=feasible,
        )
    )
    with pytest.raises(ValueError, match="candidate checks exceed"):
        compute_from_feasible_sets(request)
