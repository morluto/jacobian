"""Exact aperiodic and cyclic autocorrelation contracts."""

import json
from collections.abc import Callable
from fractions import Fraction
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationCell,
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    FiniteSequence,
    SequenceOrderShapeResult,
)
from jacobian.math.number_theory.sequences.core.operations import (
    MAX_ORDER_SHAPE_RESULT_ALLOCATIONS,
    aperiodic_autocorrelation,
    cyclic_autocorrelation,
    sequence_order_shape,
)
from jacobian.math.number_theory.sequences.core.values import MAX_SEQUENCE_LENGTH


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
    with pytest.raises(JsonSchemaValidationError):
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


def test_order_shape_accounts_for_absorbing_zeros_in_product_bounds() -> None:
    wide = CanonicalRational(num=10**20_000, den=1)
    zero = CanonicalRational(num=0, den=1)
    result = sequence_order_shape(FiniteRationalSequence(values=(wide, zero, zero)))
    assert result.log_concavity_rows[0].square == zero
    assert result.log_concavity_rows[0].neighbor_product == zero


def test_order_shape_sums_per_row_product_widths_instead_of_a_global_maximum() -> None:
    wide = CanonicalRational(num=10**16_000, den=1)
    zero = CanonicalRational(num=0, den=1)
    source = FiniteRationalSequence(values=(zero, wide) + (zero,) * 198)
    result = sequence_order_shape(source)
    assert result.log_concavity_rows[0].square.num == 10**32_000
    assert all(row.neighbor_product.num == 0 for row in result.log_concavity_rows)


def test_order_shape_admits_cross_cancelled_neighbor_products() -> None:
    tall = CanonicalRational(num=10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1), den=1)
    unit = CanonicalRational(num=1, den=tall.num)
    zero = CanonicalRational(num=0, den=1)
    result = sequence_order_shape(FiniteRationalSequence(values=(unit, zero, tall)))
    assert result.log_concavity_rows[0].neighbor_product == CanonicalRational(
        num=1, den=1
    )
    assert result.log_concavity_rows[0].square == zero


def test_rational_sequence_rejects_oversized_length_before_expansion() -> None:
    payload = {
        "domain": "rational",
        "values": [0] * (MAX_SEQUENCE_LENGTH + 1),
    }
    with pytest.raises(Exception, match="length"):
        FiniteRationalSequence.model_validate(payload)


def test_rational_sequence_rejects_oversized_integer_strings_before_parse() -> None:
    digits = "1" * (MAX_CANONICAL_INTEGER_DIGITS + 1)
    with pytest.raises(Exception, match="digit"):
        FiniteRationalSequence.model_validate_json(
            '{"domain":"rational","values":["' + digits + '"]}'
        )


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
    assert "oneOf" in validation["properties"]["values"]["items"]


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

def test_rational_coefficients_retain_exact_domain_and_source() -> None:
    source = FiniteRationalSequence(
        values=(
            CanonicalRational(num=1, den=2),
            CanonicalRational(num=1, den=3),
            CanonicalRational(num=-1, den=2),
        )
    )
    result = aperiodic_autocorrelation(source)

    assert result.source is source
    assert values(result) == [
        (-2, CanonicalRational(num=-1, den=4)),
        (-1, CanonicalRational(num=0, den=1)),
        (0, CanonicalRational(num=11, den=18)),
        (1, CanonicalRational(num=0, den=1)),
        (2, CanonicalRational(num=-1, den=4)),
    ]



def test_serialized_integer_result_retains_integer_source_domain() -> None:
    result = aperiodic_autocorrelation(FiniteIntegerSequence(values=(1, 2, 3)))

    restored = AutocorrelationResult.model_validate_json(result.model_dump_json())

    assert isinstance(restored.source, FiniteIntegerSequence)
    assert all(isinstance(cell.value, int) for cell in restored.cells)



def test_serialized_rational_integer_entries_retain_rational_domain() -> None:
    source = FiniteRationalSequence.model_validate_json(
        json.dumps({"values": ["1", "2", "3"]})
    )

    restored = AutocorrelationResult.model_validate_json(
        aperiodic_autocorrelation(source).model_dump_json()
    )

    assert isinstance(restored.source, FiniteRationalSequence)
    assert all(isinstance(cell.value, CanonicalRational) for cell in restored.cells)



def test_serialized_empty_rational_source_retains_rational_domain() -> None:
    source = FiniteRationalSequence(values=())

    restored = AutocorrelationResult.model_validate_json(
        aperiodic_autocorrelation(source).model_dump_json()
    )

    assert isinstance(restored.source, FiniteRationalSequence)



def test_rational_wire_entries_accept_full_width_integer_strings() -> None:
    value = "1" * 4_301

    source = FiniteRationalSequence.model_validate_json(json.dumps({"values": [value]}))

    assert source.values[0].num == parse_canonical_integer(value)
    assert source.values[0].den == 1



def test_result_rejects_noncanonical_lag_axis_without_recomputing_coefficients() -> (
    None
):
    with pytest.raises(ValidationError, match="invalid_lag_axis"):
        AutocorrelationResult(
            convention="cyclic",
            source=FiniteIntegerSequence(values=(1, 2)),
            cells=(
                AutocorrelationCell(lag=0, value=5),
                AutocorrelationCell(lag=2, value=5),
            ),
        )



def test_rational_wire_entries_normalize_integer_strings() -> None:
    source = FiniteRationalSequence.model_validate_json(
        '{"values":["1",{"num":"1","den":"2"}]}'
    )

    assert source.values == (
        CanonicalRational(num=1, den=1),
        CanonicalRational(num=1, den=2),
    )



def test_complex_entries_are_rejected_by_real_rational_contract() -> None:
    with pytest.raises(ValidationError):
        FiniteRationalSequence.model_validate({"values": [{"real": 1, "imaginary": 2}]})



def test_aperiodic_matches_defining_sum_for_signed_rational_lags() -> None:
    source = FiniteRationalSequence(
        values=tuple(
            CanonicalRational.from_fraction(Fraction(value, 5))
            for value in (2, -1, 3, 0)
        )
    )
    expected = {
        lag: sum(
            Fraction(source.values[index].num, source.values[index].den)
            * Fraction(
                source.values[index + lag].num,
                source.values[index + lag].den,
            )
            for index in range(len(source.values) - lag)
        )
        for lag in range(len(source.values))
    }
    expected.update({-lag: value for lag, value in expected.items() if lag})

    result = aperiodic_autocorrelation(source)
    assert [cell.lag for cell in result.cells] == list(range(-3, 4))
    actual = []
    for cell in result.cells:
        assert isinstance(cell.value, CanonicalRational)
        actual.append(cell.value.as_fraction())
    assert actual == [expected[lag] for lag in range(-3, 4)]



def test_reversing_source_preserves_aperiodic_profile() -> None:
    source = FiniteIntegerSequence(values=(2, -1, 3, 0))

    result = aperiodic_autocorrelation(source)
    reversed_result = aperiodic_autocorrelation(
        FiniteIntegerSequence(values=tuple(reversed(source.values)))
    )

    assert values(reversed_result) == values(result)



def test_rational_autocorrelation_admission_counts_both_output_components() -> None:
    width = 5_000
    numerator = parse_canonical_integer("1" + "0" * (width - 1))
    denominator = parse_canonical_integer("1" + "2" * (width - 1))
    source = FiniteRationalSequence(
        values=(CanonicalRational.from_integer_ratio(numerator, denominator),) * 126
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="autocorrelation output exceeds the exact representation bound",
    ):
        aperiodic_autocorrelation(source)



def test_catalog_accepts_serialized_integer_sequence_source() -> None:
    source = FiniteIntegerSequence(values=(1, 2, 3))
    payload = json.loads(source.model_dump_json())
    result = invoke_operation(
        "sequence.autocorrelation.aperiodic.compute",
        payload,
        Catalog.open(),
    )
    native = aperiodic_autocorrelation(source)
    assert result.output == native.model_dump(mode="json")
    restored = AutocorrelationResult.model_validate_json(json.dumps(result.output))
    assert isinstance(restored.source, FiniteIntegerSequence)
    assert restored.source.values == (1, 2, 3)



def test_wide_rational_cyclic_work_is_rejected_before_kernel() -> None:
    denominator = 10**15_999
    source = FiniteRationalSequence(
        values=(CanonicalRational(num=1, den=denominator),) * 153
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        cyclic_autocorrelation(source)



def test_mixed_denominator_widths_are_preflighted_without_scaled_copies() -> None:
    wide = CanonicalRational(num=1, den=10**15_999)
    ones = (CanonicalRational(num=1, den=1),) * 2_000
    source = FiniteRationalSequence(values=(wide, *ones))
    with pytest.raises(
        (OperationResourceAdmissionError, OperationDomainValidationError)
    ):
        cyclic_autocorrelation(source)



def test_oversized_integer_wire_entries_are_rejected_before_parsing() -> None:
    payload = {
        "domain": "rational",
        "values": ["1" * (MAX_CANONICAL_INTEGER_DIGITS + 1)],
    }
    with pytest.raises(ValidationError, match="digit"):
        FiniteRationalSequence.model_validate(payload)



def test_autocorrelation_catalog_schema_registers_canonical_rational_defs() -> None:
    schema = FiniteSequence.model_json_schema()
    assert "CanonicalRational" in json.dumps(schema)
    catalog = Catalog.open()
    descriptor = next(
        operation
        for operation in catalog.snapshot().operations
        if operation.operation_id == "sequence.autocorrelation.aperiodic.compute"
    )
    assert "CanonicalRational" in json.dumps(descriptor.input_schema)
