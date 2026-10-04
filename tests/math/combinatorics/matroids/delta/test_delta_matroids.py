"""Behavioral tests for finite delta-matroid recognition."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from tests.error_assertions import error_code

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids import delta as delta_matroids
from jacobian.math.combinatorics.matroids.delta import FiniteDeltaMatroid
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceProfileRequest,
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidTwistResult,
    DeltaMatroidWidthRequest,
)
from jacobian.math.combinatorics.matroids.delta._tools import (
    TOOLS,
    _from_feasible_sets,
    _twist,
    _width,
)
from jacobian.math.combinatorics.matroids.delta.extra import (
    BinaryMatrixResult,
    BinarySymmetricMatrix,
    DeltaMatroidTwistPolynomialRequest,
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import (
    binary,
    loop_complement,
    twist_polynomial,
)
from jacobian.math.combinatorics.matroids.delta.relabel import (
    MAX_DELTA_RELABEL_GROUND,
    MAX_DELTA_RELABEL_OUTPUT_CELLS,
    MAX_DELTA_RELABEL_TRANSPORT_WORK,
    MAX_DELTA_RELABEL_WORK,
    DeltaMatroidRelabelling,
    DeltaMatroidRelabelRequest,
    relabel,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
)


def _two_element_delta_matroid(*, scrambled: bool = False) -> FiniteFeasibleSetSystem:
    feasible = ((0,), (), (0, 1), (1,)) if scrambled else ((), (0,), (0, 1), (1,))
    return FiniteFeasibleSetSystem(ground=("a", "b"), feasible=feasible)


def test_binary_identity_matrix_constructs_canonical_delta_matroid() -> None:
    result = binary(BinarySymmetricMatrix(ground=("a", "b"), entries=((1, 0), (0, 1))))
    assert result.delta_matroid.feasible == ((), (0,), (0, 1), (1,))


def _gf2_nonsingular_by_row_reduction(matrix: tuple[tuple[int, ...], ...]) -> bool:
    """Independent small-matrix oracle for principal-minor nonsingularity."""

    rows = [list(row) for row in matrix]
    rank = 0
    for column in range(len(rows)):
        pivot = next((row for row in range(rank, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row in range(rank + 1, len(rows)):
            if rows[row][column]:
                rows[row] = [
                    left ^ right
                    for left, right in zip(rows[row], rows[rank], strict=True)
                ]
        rank += 1
    return rank == len(rows)


def test_binary_principal_minors_match_independent_gf2_oracle_through_order_four() -> (
    None
):
    """Exercise all symmetric matrices through order four (1,099 matrices)."""

    for order in range(5):
        upper_positions = tuple(
            (row, column) for row in range(order) for column in range(row, order)
        )
        for matrix_mask in range(1 << len(upper_positions)):
            rows = [[0] * order for _ in range(order)]
            for bit, (row, column) in enumerate(upper_positions):
                value = matrix_mask >> bit & 1
                rows[row][column] = rows[column][row] = value
            matrix = tuple(tuple(row) for row in rows)
            result = binary(
                BinarySymmetricMatrix(
                    ground=tuple(f"e{i}" for i in range(order)), entries=matrix
                )
            )
            expected = []
            for subset_mask in range(1 << order):
                indices = tuple(i for i in range(order) if subset_mask >> i & 1)
                principal = tuple(tuple(matrix[i][j] for j in indices) for i in indices)
                if _gf2_nonsingular_by_row_reduction(principal):
                    expected.append(indices)
            assert result.delta_matroid.feasible == tuple(sorted(expected))
            assert result.matrix.entries == matrix


def test_catalog_contains_only_audited_agent_outcome() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "delta_matroid.direct_sum.compute",
        "delta_matroid.from_feasible_sets.compute",
        "delta_matroid.distance.compute",
        "delta_matroid.twist.compute",
        "delta_matroid.width.compute",
        "delta_matroid.distance_profile.compute",
        "delta_matroid.binary.from_matrix_twist.compute",
        "delta_matroid.dual.compute",
        "delta_matroid.minor.compute",
        "delta_matroid.from_binary_matrix.compute",
        "delta_matroid.twist_polynomial.compute",
        "delta_matroid.binary_loop_complement.compute",
        "delta_matroid.twist_width_profile.compute",
        "delta_matroid.feasible_size_profile.compute",
        "delta_matroid.relabel.compute",
        "delta_matroid.distance_interlace_polynomial.compute",
    }


def test_relabel_permutation_transports_feasible_sets_and_source_maps() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1)))

    result = relabel(source, ("B", "A"), (1, 0))

    assert result.source == source
    assert result.relabelled.ground == ("B", "A")
    assert result.relabelled.feasible == ((), (0, 1), (1,))
    assert result.target_to_source == (1, 0)
    assert result.source_to_target == (1, 0)
    assert (
        DeltaMatroidRelabelling.model_validate_json(result.model_dump_json()) == result
    )


def test_relabel_composes_to_identity_and_handles_empty_ground() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1)))
    swapped = relabel(source, ("B", "A"), (1, 0))
    restored = relabel(swapped.relabelled, ("a", "b"), (1, 0))
    assert restored.relabelled == source
    assert restored.source_to_target == (1, 0)

    empty = FiniteDeltaMatroid(ground=(), feasible=((),))
    assert relabel(empty, (), ()).relabelled == empty


def test_relabel_request_map_rejects_boolean_indices_in_python_and_json() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((),))
    request = DeltaMatroidRelabelRequest(
        delta_matroid=source,
        target_ground=("A", "B"),
        target_to_source=(0, 1),
    )
    python_payload = request.model_dump(mode="python")
    python_payload["target_to_source"] = (True, False)
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelRequest.model_validate(python_payload)

    json_payload = request.model_dump(mode="json")
    json_payload["target_to_source"] = [True, False]
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelRequest.model_validate_json(json.dumps(json_payload))


@pytest.mark.parametrize("field", ["target_to_source", "source_to_target"])
def test_relabel_result_maps_reject_boolean_indices_in_python_and_json(
    field: str,
) -> None:
    result = relabel(
        FiniteDeltaMatroid(ground=("a", "b"), feasible=((),)),
        ("A", "B"),
        (0, 1),
    )
    python_payload = result.model_dump(mode="python")
    python_payload[field] = (True, False)
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelling.model_validate(python_payload)

    json_payload = result.model_dump(mode="json")
    json_payload[field] = [True, False]
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelling.model_validate_json(json.dumps(json_payload))


def test_relabel_requires_a_bijection_and_distinct_bounded_labels() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((),))
    with pytest.raises(ValueError) as exc_info:
        DeltaMatroidRelabelRequest(
            delta_matroid=source,
            target_ground=("A", "B"),
            target_to_source=(0, 0),
        )
    assert error_code(exc_info.value) == "delta_matroid.relabel_map_bijection"
    with pytest.raises(ValueError) as exc_info:
        DeltaMatroidRelabelRequest(
            delta_matroid=source,
            target_ground=("A", "A"),
            target_to_source=(0, 1),
        )
    assert error_code(exc_info.value) == "delta_matroid.relabel_ground_unique"
    admitted_labels = ("A" * 2047, "B")
    accepted = relabel(source, admitted_labels, (0, 1))
    assert accepted.relabelled.ground == admitted_labels
    from jacobian.catalog.models import OperationResourceAdmissionError

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        relabel(source, ("A" * 2048, "B"), (0, 1))
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.relabel_target_bytes"


def test_relabel_schema_advertises_runtime_admission_limits() -> None:
    schema = DeltaMatroidRelabelRequest.model_json_schema()
    limits = schema["admission_limits"]
    assert limits["max_ground_elements"] == MAX_DELTA_RELABEL_GROUND
    assert limits["max_feasible_set_memberships"] == 16_384
    assert limits["max_transport_work_units"] == MAX_DELTA_RELABEL_TRANSPORT_WORK
    assert limits["max_total_work_units"] == MAX_DELTA_RELABEL_WORK
    assert MAX_DELTA_RELABEL_WORK == (
        MAX_DELTA_RELABEL_TRANSPORT_WORK
        + 2 * MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
        + MAX_DELTA_RELABEL_OUTPUT_CELLS
    )
    assert limits["max_output_cells"] == MAX_DELTA_RELABEL_OUTPUT_CELLS
    assert schema["properties"]["target_ground"]["maxItems"] == (
        MAX_DELTA_RELABEL_GROUND
    )
    assert schema["properties"]["target_to_source"]["maxItems"] == (
        MAX_DELTA_RELABEL_GROUND
    )


def test_relabel_rejects_forged_source_that_violates_exchange() -> None:
    forged = FiniteDeltaMatroid.model_construct(
        ground=("a", "b", "c"), feasible=((), (0, 1), (2,))
    )
    with pytest.raises(ValueError) as exc_info:
        relabel(forged, ("A", "B", "C"), (0, 1, 2))
    assert error_code(exc_info.value) == "delta_matroid.source_not_delta"


def test_relabel_canonicalizes_a_forged_source_before_returning() -> None:
    forged = FiniteDeltaMatroid.model_construct(
        ground=["a", "b"], feasible=((), (0,), (0, 1))
    )

    result = relabel(forged, ("A", "B"), (0, 1))

    assert result.source is not forged
    assert result.source.ground == ("a", "b")
    assert isinstance(result.source.ground, tuple)
    assert isinstance(result.source.feasible, tuple)
    assert result.source == FiniteDeltaMatroid(
        ground=("a", "b"), feasible=((), (0,), (0, 1))
    )


def test_relabel_classifies_non_utf8_source_labels_as_domain_errors() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    forged = FiniteDeltaMatroid.model_construct(
        ground=("bad\ud800",),
        feasible=((),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        relabel(forged, ("good",), (0,))
    assert error.value.errors()[0]["type"] == "delta_matroid.labels_not_utf8"


def test_twist_request_publishes_admission_limits() -> None:
    schema = DeltaMatroidTwistRequest.model_json_schema()
    limits = schema["admission_limits"]
    assert limits["max_feasible_set_memberships"] == 16_384
    assert limits["max_ground_label_utf8_bytes"] == 2_048
    assert limits["max_symmetric_exchange_candidate_checks_per_replay"] == 250_000
    description = schema["properties"]["delta_matroid"]["description"]
    assert "16384" in description.replace(",", "")
    assert "2048" in description.replace(",", "")
    assert "250000" in description.replace(",", "")
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (1,)))
    request = DeltaMatroidTwistRequest(delta_matroid=source, subset=(0,))
    result = _twist(request)

    assert result.delta_matroid == source
    assert result.subset == (0,)
    assert result.twisted == FiniteDeltaMatroid(
        ground=("a", "b"), feasible=((), (0,), (0, 1))
    )
    assert (
        DeltaMatroidTwistResult.model_validate_json(result.model_dump_json()) == result
    )
    assert len(result.twisted.feasible) == len(source.feasible)
    assert sum(map(len, result.twisted.feasible)) <= 16_384
    twist_tool = next(
        tool for tool in TOOLS if tool.operation_id == "delta_matroid.twist.compute"
    )
    assert twist_tool.result_type is DeltaMatroidTwistResult
    assert (
        DeltaMatroidTwistRequest.model_validate_json(request.model_dump_json())
        == request
    )


def test_twist_rejects_forged_non_delta_source() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=((), (0, 1), (2,)))
    with pytest.raises(ValueError) as exc_info:
        _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(1,)))
    assert error_code(exc_info.value) == "delta_matroid.source_not_valid"


def test_width_is_exact_and_preserves_source_binding() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))
    result = _width(DeltaMatroidWidthRequest(delta_matroid=source))
    assert result.width == 2
    assert result.delta_matroid == source
    assert type(result.model_dump()["width"]) is int


def test_complete_feasible_family_constructs_canonical_delta_matroid() -> None:
    result = _from_feasible_sets(
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

    wire_result = _from_feasible_sets(
        DeltaMatroidFromFeasibleSetsRequest(system=system)
    )
    assert wire_result == native_result
    assert (
        DeltaMatroidRecognitionResult.model_validate_json(wire_result.model_dump_json())
        == wire_result
    )


def test_distance_profile_preflights_ground_before_nested_carrier_parsing() -> None:
    payload = {
        "delta_matroid": {
            "ground": [f"e{index}" for index in range(13)],
            "feasible": [["not-an-index"] for _ in range(20_000)],
        }
    }
    with pytest.raises(ValidationError) as error:
        DeltaMatroidDistanceProfileRequest.model_validate(payload)
    assert error.value.errors()[0]["type"] == (
        "delta_matroid.distance_profile_states_exceeded"
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
    claim = DeltaMatroidRecognitionResult.model_validate(forged)
    assert claim
    assert not delta_matroids.verify_from_feasible_sets(claim)


def test_serialized_recognition_claim_is_verified_against_its_source() -> None:
    result = delta_matroids.from_feasible_sets(_two_element_delta_matroid())
    assert delta_matroids.verify_from_feasible_sets(
        DeltaMatroidRecognitionResult.model_validate_json(result.model_dump_json())
    )


def test_delta_matroid_value_deserialization_does_not_replay_exchange() -> None:
    feasible = ((), (0, 1), (2,))
    value = FiniteDeltaMatroid(
        ground=("a", "b", "c"),
        feasible=feasible,
    )

    result = delta_matroids.from_feasible_sets(
        FiniteFeasibleSetSystem(ground=value.ground, feasible=value.feasible)
    )
    assert result.status == "NOT_A_DELTA_MATROID"
    assert result.obstruction is not None


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

    wire_result = _from_feasible_sets(
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
    with pytest.raises(ValueError) as exc_info:
        _from_feasible_sets(request)
    assert error_code(exc_info.value) == "delta_matroid.label_bytes_exceeded"


def test_non_utf8_representable_ground_labels_are_rejected_not_host_errors() -> None:
    # An unpaired surrogate is structurally well formed for the shared carrier
    # but has no UTF-8 byte length, so admission must reject it with a
    # controlled validation error instead of leaking UnicodeEncodeError.
    system = FiniteFeasibleSetSystem(ground=("\ud800",), feasible=((),))

    with pytest.raises(ValueError, match="UTF-8-representable"):
        delta_matroids.from_feasible_sets(system)

    request = DeltaMatroidFromFeasibleSetsRequest(system=system)
    with pytest.raises(ValueError) as exc_info:
        _from_feasible_sets(request)
    assert error_code(exc_info.value) == "delta_matroid.labels_not_utf8"

    assert FiniteDeltaMatroid(ground=("\ud800",), feasible=((),))


def test_request_schema_exposes_every_delta_specific_admission_limit() -> None:
    schema = DeltaMatroidFromFeasibleSetsRequest.model_json_schema()

    assert schema["admission_limits"] == {
        "max_feasible_set_memberships": 16_384,
        "max_ground_label_utf8_bytes": 2_048,
        "max_symmetric_exchange_candidate_checks_per_replay": 250_000,
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


def test_exchange_envelope_rejects_wide_families_without_a_row_cap() -> None:
    # Six hundred distinct pairs fit the membership envelope but exceed
    # the symmetric-exchange candidate-work bound.
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
    with pytest.raises(ValueError) as exc_info:
        _from_feasible_sets(request)
    assert error_code(exc_info.value) == "delta_matroid.candidate_work_exceeded"


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
    with pytest.raises(ValueError) as exc_info:
        _from_feasible_sets(request)
    assert error_code(exc_info.value) == "delta_matroid.candidate_work_exceeded"


def test_dense_twist_composes_with_width_and_inverse_twist() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(33)),
        feasible=((), *tuple((i,) for i in range(33))),
    )
    subset = tuple(range(33))
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=subset))
    restored = DeltaMatroidTwistResult.model_validate_json(
        result.model_dump_json()
    ).twisted
    assert sum(map(len, restored.feasible)) == 1089
    assert _width(DeltaMatroidWidthRequest(delta_matroid=restored)).width == 1
    assert (
        _twist(DeltaMatroidTwistRequest(delta_matroid=restored, subset=subset)).twisted
        == source
    )


def test_twist_output_limit_remains_a_resource_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.combinatorics.matroids.delta import operations

    def fail_if_replayed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exchange replayed before output preflight")

    monkeypatch.setattr(
        operations, "first_symmetric_exchange_obstruction", fail_if_replayed
    )
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(129)),
        feasible=((), *tuple((i,) for i in range(129))),
    )
    with pytest.raises(OperationResourceAdmissionError, match="memberships exceed"):
        _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=tuple(range(129))))


def test_native_transforms_accept_canonical_mathematical_values() -> None:
    from jacobian.math.combinatorics.matroids.delta import twist, width

    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (1,)))
    result = twist(source, (0,))
    bound = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(0,)))
    assert result == bound.twisted
    assert width(result) == _width(DeltaMatroidWidthRequest(delta_matroid=result)).width
    assert twist(result, (0,)) == source


def test_twist_result_binds_exact_source_axis_and_row_bijection() -> None:
    source = FiniteDeltaMatroid(
        ground=("left", "middle", "right"),
        feasible=((), (0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (2,)),
    )
    subset = (0, 2)
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=subset))

    assert result.delta_matroid.ground == result.twisted.ground == source.ground
    assert set(result.twisted.feasible) == {
        tuple(sorted(set(row) ^ set(subset))) for row in source.feasible
    }
    assert len(result.twisted.feasible) == len(source.feasible)


def test_twist_result_composition_uses_symmetric_difference() -> None:
    source = FiniteDeltaMatroid(
        ground=("a", "b", "c"), feasible=((), (0,), (0, 1), (1,))
    )
    first = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(0, 2)))
    second = _twist(
        DeltaMatroidTwistRequest(delta_matroid=first.twisted, subset=(1, 2))
    )
    direct = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(0, 1)))

    assert second.twisted == direct.twisted


def test_empty_twist_result_roundtrips_and_is_identity() -> None:
    source = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source))

    assert result.subset == ()
    assert result.twisted == source
    assert (
        DeltaMatroidTwistResult.model_validate_json(result.model_dump_json()) == result
    )


def test_twist_result_json_rejects_a_different_ground_axis() -> None:
    import json

    source = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source))
    payload = json.loads(result.model_dump_json())
    payload["twisted"]["ground"] = ["different"]

    with pytest.raises(ValueError) as exc_info:
        DeltaMatroidTwistResult.model_validate_json(json.dumps(payload))
    assert error_code(exc_info.value) == "delta_matroid.twist_ground_axis"


def test_twist_result_json_rejects_changed_feasible_family_cardinality() -> None:
    import json

    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (1,)))
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(0,)))
    payload = json.loads(result.model_dump_json())
    payload["twisted"]["feasible"] = [[], [0]]

    with pytest.raises(ValueError) as exc_info:
        DeltaMatroidTwistResult.model_validate_json(json.dumps(payload))
    assert error_code(exc_info.value) == "delta_matroid.twist_family_cardinality"


def test_twist_result_membership_admission_has_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.delta._tools as tools_module
    import jacobian.math.combinatorics.matroids.delta.operations as operations_module
    from jacobian.catalog.models import OperationResourceAdmissionError

    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (1,)))
    request = DeltaMatroidTwistRequest(delta_matroid=source, subset=(0,))
    # The twisted family has rows {0}, {}, and {0, 1}: exactly three
    # memberships. Admission must accept that boundary and reject one below it.
    monkeypatch.setattr(operations_module, "MAX_DELTA_MEMBERSHIPS", 3)
    assert tools_module._twist(request).twisted.feasible == ((), (0,), (0, 1))

    monkeypatch.setattr(operations_module, "MAX_DELTA_MEMBERSHIPS", 2)
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        tools_module._twist(request)


def test_even_subset_width_uses_linear_admission() -> None:
    from jacobian.math.combinatorics.matroids.delta import width

    source = FiniteDeltaMatroid(
        ground=tuple(f"e{index}" for index in range(8)),
        feasible=tuple(
            sorted(
                tuple(bit for bit in range(8) if (index >> bit) & 1)
                for index in range(256)
                if index.bit_count() % 2 == 0
            )
        ),
    )
    assert width(source) == 8


def test_feasible_size_profile_is_complete_ascending_histogram() -> None:
    from jacobian.math.combinatorics.matroids.delta import feasible_size_profile

    source = FiniteDeltaMatroid(
        ground=("a", "b", "c"),
        feasible=((), (0,), (0, 1), (1,)),
    )

    profile = feasible_size_profile(source)

    assert profile.ground == source.ground
    assert profile.counts_by_size == (1, 2, 1, 0)
    assert sum(profile.counts_by_size) == len(source.feasible)


def test_feasible_size_profile_preflights_output_before_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.combinatorics.matroids.delta import (
        extra_ops,
        feasible_size_profile,
    )

    def fail_if_replayed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exchange validation ran before output admission")

    monkeypatch.setattr(extra_ops, "_check", fail_if_replayed)
    monkeypatch.setattr(extra_ops, "MAX_FEASIBLE_SIZE_PROFILE_ENTRIES", 2)
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((),))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        feasible_size_profile(source)
    assert (
        exc_info.value.errors()[0]["type"]
        == "delta_matroid.feasible_size_profile_output"
    )


def test_feasible_size_profile_preflights_memberships_before_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.combinatorics.matroids.delta import (
        extra_ops,
        feasible_size_profile,
    )

    ground = tuple(f"e{index}" for index in range(14))
    feasible = tuple(
        sorted(
            tuple(element for element in range(14) if mask & (1 << element))
            for mask in range(1 << 14)
        )
    )
    source = FiniteDeltaMatroid(ground=ground, feasible=feasible)

    def fail_if_revalidated(_value: object) -> FiniteDeltaMatroid:
        raise AssertionError("profile copied the oversized family before admission")

    monkeypatch.setattr(extra_ops, "_admit_delta", fail_if_revalidated)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        feasible_size_profile(source)
    assert (
        exc_info.value.errors()[0]["type"] == "delta_matroid.feasible_size_profile_work"
    )


def test_feasible_size_profile_does_not_replay_delta_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.matroids.delta import (
        extra_ops,
        feasible_size_profile,
    )

    def fail_if_replayed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the profile does not depend on symmetric exchange")

    monkeypatch.setattr(extra_ops, "_check", fail_if_replayed)
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))

    assert feasible_size_profile(source).counts_by_size == (1, 2, 1)


def test_width_ignores_recognition_label_envelope() -> None:
    from jacobian.math.combinatorics.matroids.delta import width

    source = FiniteDeltaMatroid(ground=("a" * 2049,), feasible=((),))
    assert width(source) == 0


def test_width_rejects_a_forged_malformed_source() -> None:
    """A constructed source with duplicate row indices is not a valid value."""

    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.combinatorics.matroids.delta import width

    forged = FiniteDeltaMatroid.model_construct(
        ground=("a", "b"), feasible=((), (0, 0))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        width(forged)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_width_rejects_an_empty_forged_source() -> None:
    """An empty feasible family is not a delta-matroid and has no width."""

    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.combinatorics.matroids.delta import width

    forged = FiniteDeltaMatroid.model_construct(ground=(), feasible=())
    with pytest.raises(OperationDomainValidationError) as error:
        width(forged)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"
    assert "at least one feasible set" in str(error.value)


def test_minor_compacts_axes_and_applies_deletion_semantics() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))
    from jacobian.math.combinatorics.matroids.delta.extra_ops import minor

    assert minor(source, delete=(0,)) == FiniteDeltaMatroid(
        ground=("b",), feasible=((), (0,))
    )


def test_minor_rejects_model_constructed_missing_axes() -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.combinatorics.matroids.delta.extra_ops import minor

    forged = FiniteDeltaMatroid.model_construct(feasible=((),))
    with pytest.raises(OperationDomainValidationError):
        minor(forged)


def test_binary_result_decoding_does_not_replay_principal_minors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.delta.extra_ops as extra_ops

    result = binary(BinarySymmetricMatrix(ground=("a", "b"), entries=((0, 0), (0, 0))))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result validation must not replay principal minors")

    monkeypatch.setattr(extra_ops, "_det2", fail)

    assert BinaryMatrixResult.model_validate_json(result.model_dump_json()) == result


def test_binary_loop_complement_matches_independent_feasible_family_toggle() -> None:
    """Diagonal toggles agree with the delta-matroid loop-complement axiom."""
    for order in range(4):
        upper = tuple((i, j) for i in range(order) for j in range(i, order))
        for matrix_mask in range(1 << len(upper)):
            rows = [[0] * order for _ in range(order)]
            for bit, (i, j) in enumerate(upper):
                rows[i][j] = rows[j][i] = matrix_mask >> bit & 1
            entries = tuple(tuple(row) for row in rows)
            source = BinarySymmetricMatrix(
                ground=tuple(f"e{i}" for i in range(order)), entries=entries
            )
            original = set(binary(source).delta_matroid.feasible)
            for subset_mask in range(1 << order):
                subset = tuple(i for i in range(order) if subset_mask >> i & 1)
                expected = set(original)
                for element in subset:
                    for feasible in tuple(expected):
                        if element not in feasible:
                            toggled = tuple(sorted((*feasible, element)))
                            if toggled in expected:
                                expected.remove(toggled)
                            else:
                                expected.add(toggled)
                result = loop_complement(source, subset)
                assert result.source == source
                assert result.result.matrix.ground == source.ground
                assert result.result.delta_matroid.feasible == tuple(sorted(expected))


def test_binary_loop_complement_rejects_noncanonical_subset() -> None:
    from pydantic import ValidationError

    from jacobian.math.combinatorics.matroids.delta.extra import (
        BinaryLoopComplementRequest,
    )

    matrix = BinarySymmetricMatrix(ground=("a", "b"), entries=((0, 0), (0, 0)))
    with pytest.raises(ValidationError):
        BinaryLoopComplementRequest(matrix=matrix, subset=(1, 0))


def test_extra_operation_rejects_forged_non_delta_source() -> None:
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.combinatorics.matroids.delta._tools import _run_dual

    forged = FiniteDeltaMatroid.model_construct(
        ground=("a", "b", "c"), feasible=((), (0, 1), (2,))
    )
    with pytest.raises(OperationDomainValidationError):
        _run_dual(type("Request", (), {"delta_matroid": forged})())


def _satisfies_symmetric_exchange(feasible: set[frozenset[int]]) -> bool:
    for left in feasible:
        for right in feasible:
            difference = left ^ right
            for element in difference:
                if not any(
                    (left ^ {element, candidate}) in feasible
                    for candidate in difference
                ):
                    return False
    return True


def _oracle_twist_polynomial(n: int, feasible: set[frozenset[int]]) -> tuple[int, ...]:
    coefficients = [0] * (n + 1)
    for twist_mask in range(1 << n):
        twist = frozenset(i for i in range(n) if twist_mask >> i & 1)
        sizes = tuple(len(row ^ twist) for row in feasible)
        coefficients[max(sizes) - min(sizes)] += 1
    return tuple(coefficients)


def _descending_nonzero_polynomial(histogram: tuple[int, ...]) -> tuple[int, ...]:
    descending = tuple(reversed(histogram))
    while len(descending) > 1 and descending[0] == 0:
        descending = descending[1:]
    return descending


def test_twist_polynomial_matches_independent_exhaustive_small_oracle() -> None:
    # Enumerate every nonempty feasible family through a three-element ground
    # and independently apply symmetric exchange and the twist definition.
    for n in range(4):
        subsets = tuple(
            frozenset(i for i in range(n) if subset_mask >> i & 1)
            for subset_mask in range(1 << n)
        )
        for family_mask in range(1, 1 << len(subsets)):
            feasible = {
                subset
                for index, subset in enumerate(subsets)
                if family_mask >> index & 1
            }
            if not _satisfies_symmetric_exchange(feasible):
                continue
            source = FiniteDeltaMatroid(
                ground=tuple(f"e{i}" for i in range(n)),
                feasible=tuple(sorted(tuple(sorted(row)) for row in feasible)),
            )
            result = twist_polynomial(source)
            expected_histogram = _oracle_twist_polynomial(n, feasible)
            assert result.coefficients_by_width == expected_histogram
            assert result.polynomial.coefficients == _descending_nonzero_polynomial(
                expected_histogram
            )
            assert result.ground == source.ground
            assert sum(result.coefficients_by_width) == 1 << n


def test_twist_polynomial_empty_axis_binary_composition_and_json_roundtrip() -> None:
    empty = FiniteDeltaMatroid(ground=(), feasible=((),))
    assert twist_polynomial(empty).coefficients_by_width == (1,)

    matrix_result = binary(
        BinarySymmetricMatrix(ground=("a", "b"), entries=((1, 0), (0, 1)))
    )
    # The identity presentation has every subset feasible, so all four twists
    # have width two and the polynomial is 4*z^2.
    result = twist_polynomial(matrix_result.delta_matroid)
    assert result.coefficients_by_width == (0, 0, 4)
    assert result.polynomial.coefficients == (4, 0, 0)
    assert type(result.model_validate_json(result.model_dump_json())) is type(result)

    request = DeltaMatroidTwistPolynomialRequest(delta_matroid=empty)
    assert (
        DeltaMatroidTwistPolynomialRequest.model_validate_json(
            request.model_dump_json()
        )
        == request
    )
    schema = DeltaMatroidTwistPolynomialRequest.model_json_schema()
    assert schema["admission_limits"]["max_twist_masks"] == 4_096
    assert schema["admission_limits"]["max_mask_feasible_set_evaluations"] == 262_144
    assert schema["admission_limits"]["max_ground_elements"] == 12
    assert schema["admission_limits"]["max_ground_label_codepoints"] == 1_000_000
    assert schema["admission_limits"]["max_histogram_entries"] == 13
    assert schema["admission_limits"]["max_polynomial_coefficient_digits"] == 4
    assert schema["admission_limits"]["max_source_memberships"] == 16_384
    assert schema["admission_limits"]["max_source_feasible_rows"] == 16_385
    assert "max_encoded_output_bytes" not in schema["admission_limits"]

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "delta_matroid.twist_polynomial.compute"
    )
    example_request = tool.request_type.model_validate(tool.examples[0].input)
    assert tool.run(example_request).coefficients_by_width == (0, 0, 4)


def test_twist_polynomial_rejects_before_expanding_too_many_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.delta.extra_ops as extra_ops

    too_wide = FiniteDeltaMatroid.model_construct(
        ground=tuple(f"e{i}" for i in range(13)), feasible=((),)
    )

    def fail_if_source_validation_runs(_value: object) -> None:
        raise AssertionError("oversized axis must reject before full source validation")

    monkeypatch.setattr(extra_ops, "_admit_delta", fail_if_source_validation_runs)
    with pytest.raises(OperationResourceAdmissionError):
        twist_polynomial(too_wide)


def test_twist_polynomial_request_preflights_raw_nested_ground() -> None:
    raw_request: dict[str, object] = {
        "delta_matroid": {
            "ground": ["same-label"] * 100_000,
            "feasible": [[]],
        }
    }

    with pytest.raises(ValidationError) as error:
        DeltaMatroidTwistPolynomialRequest.model_validate(raw_request)

    assert error.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_work"


def test_twist_polynomial_request_rejects_boolean_membership_indices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DeltaMatroidTwistPolynomialRequest.model_validate(
            {"delta_matroid": {"ground": ["a", "b"], "feasible": [[], [True]]}}
        )
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.membership_index_type"


def test_twist_polynomial_request_preflights_raw_feasible_memberships() -> None:
    raw_request = {
        "delta_matroid": {
            "ground": ["a"],
            "feasible": [[0]] * 16_385,
        }
    }

    with pytest.raises(ValidationError) as error:
        DeltaMatroidTwistPolynomialRequest.model_validate(raw_request)

    assert error.value.errors()[0]["type"] == "delta_matroid.memberships_exceeded"


def test_twist_polynomial_request_checks_rows_when_ground_is_malformed() -> None:
    raw_request: dict[str, object] = {
        "delta_matroid": {
            "feasible": [[]] * 16_386,
        }
    }

    with pytest.raises(ValidationError) as error:
        DeltaMatroidTwistPolynomialRequest.model_validate(raw_request)

    assert error.value.errors()[0]["type"] == "delta_matroid.memberships_exceeded"


@pytest.mark.parametrize("oversized_rows", (False, True))
def test_twist_polynomial_preflights_forged_feasible_family_before_copy(
    monkeypatch: pytest.MonkeyPatch, oversized_rows: bool
) -> None:
    import jacobian.math.combinatorics.matroids.delta.extra_ops as extra_ops

    feasible = ((),) * 16_386 if oversized_rows else ((0,) * 16_385,)
    source = FiniteDeltaMatroid.model_construct(ground=("a",), feasible=feasible)

    def fail_if_source_is_copied(_value: object) -> None:
        raise AssertionError(
            "oversized feasible family must reject before revalidation"
        )

    monkeypatch.setattr(extra_ops, "_admit_delta", fail_if_source_is_copied)
    with pytest.raises(OperationResourceAdmissionError) as error:
        twist_polynomial(source)
    assert error.value.errors()[0]["type"] == "delta_matroid.memberships_exceeded"


def test_twist_polynomial_accepts_ground_and_state_cardinality_boundary() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(12)), feasible=((),)
    )
    result = twist_polynomial(source)
    assert result.coefficients_by_width == (4_096, *(0 for _ in range(12)))
    assert result.polynomial.coefficients == (4_096,)


def test_twist_polynomial_ignores_recognition_label_envelope() -> None:
    # Labels do not enter the mask sweep, so the recognition operation's
    # 2,048-byte cap must not narrow this operation's advertised domain.
    label = "a" * 2_049
    source = FiniteDeltaMatroid(ground=(label,), feasible=((),))

    result = twist_polynomial(source)

    assert result.ground == (label,)
    assert result.coefficients_by_width == (2, 0)
    assert result.polynomial.coefficients == (2,)


def test_twist_polynomial_result_rejects_duplicate_ground_labels() -> None:
    import json

    from pydantic import ValidationError

    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))
    result = twist_polynomial(source)
    payload = result.model_dump(mode="json")
    payload["ground"] = ["a", "a"]

    with pytest.raises(ValidationError) as exc_info:
        type(result).model_validate_json(json.dumps(payload))
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_ground"


def test_twist_polynomial_result_rejects_inconsistent_wire_claims() -> None:
    import json

    from pydantic import ValidationError

    source = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))
    result = twist_polynomial(source)

    mismatched_polynomial = result.model_dump(mode="json")
    mismatched_polynomial["coefficients_by_width"] = [1, 1]
    with pytest.raises(ValidationError) as exc_info:
        type(result).model_validate_json(json.dumps(mismatched_polynomial))
    assert (
        exc_info.value.errors()[0]["type"]
        == "delta_matroid.twist_polynomial_coefficients"
    )

    nontotal_histogram = result.model_dump(mode="json")
    nontotal_histogram["coefficients_by_width"] = [0, 1]
    nontotal_histogram["polynomial"]["coefficients"] = ["1"]
    with pytest.raises(ValidationError) as exc_info:
        type(result).model_validate_json(json.dumps(nontotal_histogram))
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_total"


@pytest.mark.parametrize("histogram", ([True, True], ["1", "1"], [1.0, 1.0]))
def test_twist_polynomial_result_requires_strict_wire_histogram(
    histogram: list[object],
) -> None:
    import json

    from pydantic import ValidationError

    result = twist_polynomial(FiniteDeltaMatroid(ground=("a",), feasible=((),)))
    payload = result.model_dump(mode="json")
    payload["coefficients_by_width"] = histogram
    payload["polynomial"]["coefficients"] = ["1", "1"]
    with pytest.raises(ValidationError) as error:
        type(result).model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == "int_type"


def test_twist_polynomial_result_rejects_oversized_authored_axes() -> None:
    import json

    from pydantic import ValidationError

    result = twist_polynomial(FiniteDeltaMatroid(ground=("a",), feasible=((),)))
    payload = result.model_dump(mode="json")
    payload["ground"] = [f"e{i}" for i in range(13)]
    payload["coefficients_by_width"] = [8_192] + [0] * 13
    payload["polynomial"]["coefficients"] = ["8192"]

    with pytest.raises(ValidationError) as error:
        type(result).model_validate_json(json.dumps(payload))
    assert {issue["loc"] for issue in error.value.errors()} >= {
        ("ground",),
        ("coefficients_by_width",),
    }


@pytest.mark.parametrize(
    "coefficients",
    (["1"] + ["0"] * 13, ["9" * 1_000]),
)
def test_twist_polynomial_result_preflights_nested_polynomial_claim(
    coefficients: list[str],
) -> None:
    import json

    from pydantic import ValidationError

    result = twist_polynomial(FiniteDeltaMatroid(ground=("a",), feasible=((),)))
    payload = result.model_dump(mode="json")
    payload["polynomial"]["coefficients"] = coefficients
    with pytest.raises(ValidationError) as error:
        type(result).model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == (
        "delta_matroid.twist_polynomial_polynomial_bound"
    )


def test_twist_polynomial_result_rejects_overbudget_authored_label() -> None:
    import json

    from pydantic import ValidationError

    result = twist_polynomial(FiniteDeltaMatroid(ground=("a",), feasible=((),)))
    payload = result.model_dump(mode="json")
    payload["ground"] = ["a" * 1_000_001]
    with pytest.raises(ValidationError) as error:
        type(result).model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_labels"


def test_twist_polynomial_result_rejects_non_utf8_ground() -> None:
    from pydantic import ValidationError

    result = twist_polynomial(FiniteDeltaMatroid(ground=("a",), feasible=((),)))
    payload = result.model_dump(mode="python")
    payload["ground"] = ("\ud800",)
    with pytest.raises(ValidationError) as error:
        type(result).model_validate(payload)
    assert error.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_utf8"


def test_twist_polynomial_admits_own_native_label_budget_boundary() -> None:
    label = "a" * 1_000_000
    source = FiniteDeltaMatroid(ground=(label,), feasible=((),))

    result = twist_polynomial(source)

    assert result.coefficients_by_width == (2, 0)
    assert result.ground[0] is label


def test_twist_polynomial_rejects_label_growth_before_source_copy() -> None:
    source = FiniteDeltaMatroid(ground=("a" * 1_000_001,), feasible=((),))

    with pytest.raises(OperationResourceAdmissionError) as error:
        twist_polynomial(source)
    assert error.value.errors()[0]["type"] == "delta_matroid.twist_polynomial_labels"


def test_twist_polynomial_rejects_non_utf8_label_as_malformed_source() -> None:
    source = FiniteDeltaMatroid(ground=("\ud800",), feasible=((),))

    with pytest.raises(OperationDomainValidationError) as error:
        twist_polynomial(source)
    assert error.value.errors()[0]["type"] == "delta_matroid.labels_not_utf8"
