"""Behavioral tests for finite delta-matroid recognition."""

from __future__ import annotations

import pytest

from jacobian.math.combinatorics.greedoids import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids import delta as delta_matroids
from jacobian.math.combinatorics.matroids.delta import FiniteDeltaMatroid
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
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
)
from jacobian.math.combinatorics.matroids.delta.extra_ops import binary, loop_complement


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
        "delta_matroid.dual.compute",
        "delta_matroid.minor.compute",
        "delta_matroid.from_binary_matrix.compute",
        "delta_matroid.binary_loop_complement.compute",
        "delta_matroid.twist_width_profile.compute",
        "delta_matroid.feasible_size_profile.compute",
    }


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

    assert result == FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1)))
    assert (
        DeltaMatroidTwistRequest.model_validate_json(request.model_dump_json())
        == request
    )


def test_twist_rejects_forged_non_delta_source() -> None:
    source = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=((), (0, 1), (2,)))
    with pytest.raises(ValueError, match="source feasible family"):
        _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(1,)))


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
    with pytest.raises(ValueError, match="ground labels exceed"):
        _from_feasible_sets(request)


def test_non_utf8_representable_ground_labels_are_rejected_not_host_errors() -> None:
    # An unpaired surrogate is structurally well formed for the shared carrier
    # but has no UTF-8 byte length, so admission must reject it with a
    # controlled validation error instead of leaking UnicodeEncodeError.
    system = FiniteFeasibleSetSystem(ground=("\ud800",), feasible=((),))

    with pytest.raises(ValueError, match="UTF-8-representable"):
        delta_matroids.from_feasible_sets(system)

    request = DeltaMatroidFromFeasibleSetsRequest(system=system)
    with pytest.raises(ValueError, match="UTF-8-representable"):
        _from_feasible_sets(request)

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
    with pytest.raises(ValueError, match="candidate checks exceed"):
        _from_feasible_sets(request)


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
        _from_feasible_sets(request)


def test_dense_twist_composes_with_width_and_inverse_twist() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(33)),
        feasible=((), *tuple((i,) for i in range(33))),
    )
    subset = tuple(range(33))
    result = _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=subset))
    restored = FiniteDeltaMatroid.model_validate_json(result.model_dump_json())
    assert sum(map(len, restored.feasible)) == 1089
    assert _width(DeltaMatroidWidthRequest(delta_matroid=restored)).width == 1
    assert (
        _twist(DeltaMatroidTwistRequest(delta_matroid=restored, subset=subset))
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
    assert result == _twist(DeltaMatroidTwistRequest(delta_matroid=source, subset=(0,)))
    assert width(result) == _width(DeltaMatroidWidthRequest(delta_matroid=result)).width
    assert twist(result, (0,)) == source


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
    with pytest.raises(OperationResourceAdmissionError, match="output envelope"):
        feasible_size_profile(source)


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
