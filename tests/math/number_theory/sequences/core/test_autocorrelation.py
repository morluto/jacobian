"""Exact aperiodic and cyclic autocorrelation contracts."""

from collections.abc import Callable
from typing import cast

import pytest
from jsonschema import Draft202012Validator, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    SequenceOrderShapeResult,
)
from jacobian.math.number_theory.sequences.core.operations import (
    MAX_ORDER_SHAPE_RESULT_ALLOCATIONS,
    aperiodic_autocorrelation,
    cyclic_autocorrelation,
    sequence_order_shape,
)


def values(result: AutocorrelationResult) -> list[tuple[int, int]]:
    return [(cell.lag, cell.value) for cell in result.cells]


def rational_sequence(values: tuple[int, ...]) -> FiniteRationalSequence:
    return FiniteRationalSequence(
        values=tuple(CanonicalRational(num=value, den=1) for value in values)
    )


def test_aperiodic_uses_signed_nonwrapping_lags() -> None:
    result = aperiodic_autocorrelation(FiniteIntegerSequence(values=(1, 2, 3)))
    assert values(result) == [(-2, 3), (-1, 8), (0, 14), (1, 8), (2, 3)]
    assert result.source.values == (1, 2, 3)


def test_cyclic_uses_one_complete_residue_axis() -> None:
    result = cyclic_autocorrelation(FiniteIntegerSequence(values=(1, 2, 3)))
    assert values(result) == [(0, 14), (1, 11), (2, 11)]


def test_empty_sequence_has_empty_profiles() -> None:
    source = FiniteIntegerSequence(values=())
    assert aperiodic_autocorrelation(source).cells == ()
    assert cyclic_autocorrelation(source).cells == ()


def test_order_shape_retains_all_flat_peak_positions_and_signed_rows() -> None:
    source = rational_sequence((1, 3, 3, 2))
    result = sequence_order_shape(source)
    assert result.first_nondecreasing_violation == 2
    assert result.first_nonincreasing_violation == 0
    assert result.weak_unimodal_peak_positions == (1, 2)
    assert [
        (row.index, row.square, row.neighbor_product, row.holds)
        for row in result.log_concavity_rows
    ] == [
        (1, CanonicalRational(num=9, den=1), CanonicalRational(num=3, den=1), True),
        (2, CanonicalRational(num=9, den=1), CanonicalRational(num=6, den=1), True),
    ]


def test_order_shape_reports_internal_zero_and_signed_log_concavity() -> None:
    result = sequence_order_shape(rational_sequence((-2, 0, -3)))
    assert not result.is_nonnegative
    assert result.has_internal_zero
    assert result.log_concavity_rows[0].neighbor_product == CanonicalRational(
        num=6, den=1
    )
    assert not result.log_concavity_rows[0].holds


def test_rational_profile_is_exact_and_round_trips() -> None:
    source = FiniteRationalSequence(
        values=(
            CanonicalRational(num=1, den=2),
            CanonicalRational(num=3, den=4),
            CanonicalRational(num=1, den=2),
        )
    )
    result = sequence_order_shape(source)
    assert result.weak_unimodal_peak_positions == (1,)
    assert result.first_log_concavity_violation is None
    assert result.log_concavity_rows[0].square == CanonicalRational(num=9, den=16)
    assert result.log_concavity_rows[0].neighbor_product == CanonicalRational(
        num=1, den=4
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_rational_profile_accepts_integer_wire_entries_as_canonical_values() -> None:
    source = FiniteRationalSequence.model_validate_json(
        '{"values":["1",{"num":"3","den":"1"},"2"]}'
    )
    assert source.values == (
        CanonicalRational(num=1, den=1),
        CanonicalRational(num=3, den=1),
        CanonicalRational(num=2, den=1),
    )
    Draft202012Validator(FiniteRationalSequence.model_json_schema()).validate(
        {"values": ["1", {"num": "3", "den": "1"}, "2"]}
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(FiniteRationalSequence.model_json_schema()).validate(
            {"values": [2]}
        )
    assert FiniteRationalSequence.model_validate({"values": [2]}).values == (
        CanonicalRational(num=2, den=1),
    )


def test_rational_profile_parses_integer_wire_entries_beyond_python_digit_guard() -> (
    None
):
    integer = "7" + "1" * 4_999
    source = FiniteRationalSequence.model_validate_json(
        '{"values":["' + integer + '"]}'
    )
    assert source.values[0].den == 1
    assert format_canonical_integer(source.values[0].num) == integer


def test_rational_profile_reports_first_witnesses_and_vacuous_edges() -> None:
    result = sequence_order_shape(rational_sequence((3, 1, 2, 0, -1)))
    assert result.first_nondecreasing_violation == 0
    assert result.first_nonincreasing_violation == 1
    assert result.first_log_concavity_violation == 1
    assert result.first_negative_index == 4
    assert result.first_internal_zero_index == 3
    assert sequence_order_shape(FiniteRationalSequence(values=())).is_nonnegative
    assert (
        sequence_order_shape(
            FiniteRationalSequence(values=())
        ).weak_unimodal_peak_positions
        == ()
    )
    assert sequence_order_shape(
        FiniteRationalSequence(values=(CanonicalRational(num=2, den=1),))
    ).weak_unimodal_peak_positions == (0,)


def test_order_shape_native_guard_rejects_unrelated_integer_sequence_value() -> None:
    with pytest.raises(TypeError, match="FiniteRationalSequence"):
        sequence_order_shape(
            cast(FiniteRationalSequence, FiniteIntegerSequence(values=(1, 2)))
        )


def test_order_shape_skips_product_bounds_when_no_interior_rows() -> None:
    wide = CanonicalRational(num=10**16_384, den=1)
    result = sequence_order_shape(FiniteRationalSequence(values=(wide,)))
    assert result.log_concavity_rows == ()
    assert result.first_log_concavity_violation is None
    pair = sequence_order_shape(FiniteRationalSequence(values=(wide, wide)))
    assert pair.log_concavity_rows == ()


def test_order_shape_rejects_contradictory_log_concavity_witness() -> None:
    result = sequence_order_shape(rational_sequence((1, 1, 3)))
    forged = result.model_dump()
    forged["log_concavity_rows"][0]["holds"] = True
    with pytest.raises(ValueError, match="first log-concavity"):
        SequenceOrderShapeResult.model_validate(forged)
    forged_null = result.model_dump()
    forged_null["first_log_concavity_violation"] = None
    with pytest.raises(ValueError, match="first log-concavity"):
        SequenceOrderShapeResult.model_validate(forged_null)


def test_order_shape_rejects_contradictory_boolean_witnesses() -> None:
    result = sequence_order_shape(rational_sequence((1, 2, 1)))
    forged = result.model_dump()
    forged["is_nonnegative"] = True
    forged["first_negative_index"] = 0
    with pytest.raises(ValueError, match="nonnegativity"):
        SequenceOrderShapeResult.model_validate(forged)
    signed = sequence_order_shape(rational_sequence((-1, 0, -2)))
    forged_zero = signed.model_dump()
    forged_zero["has_internal_zero"] = False
    with pytest.raises(ValueError, match="internal-zero"):
        SequenceOrderShapeResult.model_validate(forged_zero)


def test_order_shape_serialization_schema_omits_integer_wire_alternative() -> None:
    serialized = SequenceOrderShapeResult.model_json_schema(mode="serialization")
    source_values = serialized["$defs"]["FiniteRationalSequence"]["properties"][
        "values"
    ]
    assert "anyOf" not in source_values["items"]
    validation = FiniteRationalSequence.model_json_schema(mode="validation")
    assert "anyOf" in validation["properties"]["values"]["items"]


def test_order_shape_result_checks_structure_without_replaying_values() -> None:
    result = sequence_order_shape(
        FiniteRationalSequence(
            values=tuple(CanonicalRational(num=value, den=1) for value in (1, 2, 1))
        )
    )
    forged = result.model_dump()
    forged["log_concavity_rows"][0]["index"] = 2
    with pytest.raises(ValueError, match="interior index"):
        SequenceOrderShapeResult.model_validate(forged)


def test_order_shape_result_rejects_out_of_range_monotonicity_witness() -> None:
    result = sequence_order_shape(
        FiniteRationalSequence(
            values=tuple(CanonicalRational(num=value, den=1) for value in (1, 2))
        )
    )
    forged = result.model_dump()
    forged["first_nondecreasing_violation"] = 1
    with pytest.raises(ValueError, match="adjacent source pair"):
        SequenceOrderShapeResult.model_validate(forged)


def test_order_shape_reversal_and_positive_scaling_preserve_decisions() -> None:
    source_values = (1, 3, 3, 2, 1)
    source = rational_sequence(source_values)
    scaled = rational_sequence(tuple(5 * value for value in source_values))
    result = sequence_order_shape(source)
    scaled_result = sequence_order_shape(scaled)
    assert (
        result.first_nondecreasing_violation,
        result.first_nonincreasing_violation,
        result.weak_unimodal_peak_positions,
        tuple(row.holds for row in result.log_concavity_rows),
        result.is_nonnegative,
        result.has_internal_zero,
    ) == (
        scaled_result.first_nondecreasing_violation,
        scaled_result.first_nonincreasing_violation,
        scaled_result.weak_unimodal_peak_positions,
        tuple(row.holds for row in scaled_result.log_concavity_rows),
        scaled_result.is_nonnegative,
        scaled_result.has_internal_zero,
    )

    reversed_result = sequence_order_shape(
        rational_sequence(tuple(reversed(source_values)))
    )
    assert reversed_result.weak_unimodal_peak_positions == (2, 3)
    assert tuple(row.holds for row in reversed_result.log_concavity_rows) == tuple(
        row.holds for row in reversed(result.log_concavity_rows)
    )


def test_order_shape_profiles_binomial_coefficients() -> None:
    source = rational_sequence((1, 5, 10, 10, 5, 1))
    result = sequence_order_shape(source)
    assert result.weak_unimodal_peak_positions == (2, 3)
    assert result.first_nondecreasing_violation == 3
    assert result.first_nonincreasing_violation == 0
    assert all(row.holds for row in result.log_concavity_rows)


@pytest.mark.parametrize(
    "operation", [aperiodic_autocorrelation, cyclic_autocorrelation]
)
def test_large_quadratic_autocorrelation_is_rejected(
    operation: Callable[[FiniteIntegerSequence], AutocorrelationResult],
) -> None:
    source = FiniteIntegerSequence(values=(1,) * 3_000)
    with pytest.raises(OperationResourceAdmissionError):
        operation(source)


def test_constant_sequence_peak_scan_is_linear() -> None:
    source = rational_sequence((1,) * 10_000)
    result = sequence_order_shape(source)
    assert result.weak_unimodal_peak_positions == tuple(range(10_000))


def test_order_shape_rejects_complete_profile_output_explosion() -> None:
    source = rational_sequence((1,) * 100_000)
    with pytest.raises(
        OperationResourceAdmissionError,
        match=str(MAX_ORDER_SHAPE_RESULT_ALLOCATIONS),
    ):
        sequence_order_shape(source)


def test_order_shape_admits_complete_rational_result_representation() -> None:
    source_value = CanonicalRational(num=1_234_567, den=7_654_321)
    source = FiniteRationalSequence(values=(source_value,) * 80_000)
    with pytest.raises(
        OperationDomainValidationError,
        match="result representation",
    ):
        sequence_order_shape(source)


def test_order_shape_native_and_catalog_paths_share_rational_carrier() -> None:
    source = rational_sequence((1, 3, 3, 2))
    native = sequence_order_shape(source)
    dispatched = invoke_operation(
        "sequence.order_shape.profile.compute",
        {"values": ["1", "3", "3", "2"]},
        Catalog.open(),
    )
    assert dispatched.output == native.model_dump(mode="json")
    assert isinstance(native.source, FiniteRationalSequence)
    assert all(
        isinstance(row.square, CanonicalRational)
        and isinstance(row.neighbor_product, CanonicalRational)
        for row in native.log_concavity_rows
    )
