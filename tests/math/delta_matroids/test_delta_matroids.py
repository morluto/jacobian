"""Behavioral tests for finite delta-matroid recognition."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math import delta_matroids
from jacobian.math.delta_matroids import FiniteDeltaMatroid
from jacobian.math.delta_matroids._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.delta_matroids._operations import compute_from_feasible_sets
from jacobian.math.delta_matroids._tools import TOOLS
from jacobian.math.greedoids import FiniteFeasibleSetSystem


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


def test_forged_valid_result_cannot_disconnect_value_from_source() -> None:
    result = delta_matroids.from_feasible_sets(_two_element_delta_matroid())
    forged = result.model_dump(mode="json")
    assert forged["delta_matroid"] is not None
    forged["delta_matroid"]["ground"][0] = "changed"

    with pytest.raises(ValidationError, match="canonical replay"):
        DeltaMatroidRecognitionResult.model_validate(forged)


def test_result_round_trips_and_replays_its_retained_source() -> None:
    result = delta_matroids.from_feasible_sets(_two_element_delta_matroid())
    assert (
        DeltaMatroidRecognitionResult.model_validate_json(result.model_dump_json())
        == result
    )

    forged = result.model_dump(mode="json")
    forged["source"]["ground"][0] = "changed"
    with pytest.raises(ValidationError, match="canonical replay"):
        DeltaMatroidRecognitionResult.model_validate(forged)


def test_invalid_delta_matroid_value_is_rejected_at_its_value_boundary() -> None:
    feasible = ((), (0, 1), (2,))
    with pytest.raises(ValidationError, match="symmetric exchange"):
        FiniteDeltaMatroid(
            ground=("a", "b", "c"),
            feasible=feasible,
        )


def test_request_rejects_ground_sets_outside_exchange_work_envelope() -> None:
    with pytest.raises(ValidationError, match="ground size"):
        DeltaMatroidFromFeasibleSetsRequest(
            system=FiniteFeasibleSetSystem(
                ground=tuple(f"e{index}" for index in range(17)),
                feasible=((),),
            )
        )


def test_request_rejects_exchange_candidate_space_before_axiom_replay() -> None:
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
    with pytest.raises(ValidationError, match="candidate checks"):
        DeltaMatroidFromFeasibleSetsRequest(
            system=FiniteFeasibleSetSystem(
                ground=tuple(f"e{index}" for index in range(8)),
                feasible=feasible,
            )
        )
