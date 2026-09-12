"""Exact aperiodic and cyclic autocorrelation contracts."""

import json
from collections.abc import Callable
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
)
from jacobian.math.number_theory.sequences.core.operations import (
    aperiodic_autocorrelation,
    cyclic_autocorrelation,
    sequence_order_shape,
)


def values(result: AutocorrelationResult) -> list[tuple[int, object]]:
    return [(cell.lag, cell.value) for cell in result.cells]


def test_aperiodic_uses_signed_nonwrapping_lags() -> None:
    result = aperiodic_autocorrelation(FiniteIntegerSequence(values=(1, 2, 3)))
    assert result.convention == "aperiodic"
    assert values(result) == [(-2, 3), (-1, 8), (0, 14), (1, 8), (2, 3)]
    assert result.source.values == (1, 2, 3)


def test_cyclic_uses_one_complete_residue_axis() -> None:
    result = cyclic_autocorrelation(FiniteIntegerSequence(values=(1, 2, 3)))
    assert result.convention == "cyclic"
    assert values(result) == [(0, 14), (1, 11), (2, 11)]


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


def test_result_rejects_noncanonical_lag_axis_without_recomputing_coefficients() -> (
    None
):
    with pytest.raises(ValidationError, match="invalid_lag_axis"):
        AutocorrelationResult(
            convention="cyclic",
            source=FiniteIntegerSequence(values=(1, 2)),
            cells=(
                {"lag": 0, "value": 5},
                {"lag": 2, "value": 5},
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
    assert [Fraction(cell.value.num, cell.value.den) for cell in result.cells] == [
        expected[lag] for lag in range(-3, 4)
    ]


def test_reversing_source_preserves_aperiodic_profile() -> None:
    source = FiniteIntegerSequence(values=(2, -1, 3, 0))

    result = aperiodic_autocorrelation(source)
    reversed_result = aperiodic_autocorrelation(
        FiniteIntegerSequence(values=tuple(reversed(source.values)))
    )

    assert values(reversed_result) == values(result)


def test_empty_sequence_has_empty_profiles() -> None:
    source = FiniteIntegerSequence(values=())
    assert aperiodic_autocorrelation(source).cells == ()
    assert cyclic_autocorrelation(source).cells == ()


def test_order_shape_retains_all_flat_peak_positions_and_signed_rows() -> None:
    source = FiniteIntegerSequence(values=(1, 3, 3, 2))
    result = sequence_order_shape(source)
    assert result.first_nondecreasing_violation == 2
    assert result.first_nonincreasing_violation == 0
    assert result.weak_unimodal_peak_positions == (1, 2)
    assert [
        (row.index, row.square, row.neighbor_product, row.holds)
        for row in result.log_concavity_rows
    ] == [
        (1, 9, 3, True),
        (2, 9, 6, True),
    ]


def test_order_shape_reports_internal_zero_and_signed_log_concavity() -> None:
    result = sequence_order_shape(FiniteIntegerSequence(values=(-2, 0, -3)))
    assert not result.is_nonnegative
    assert result.has_internal_zero
    assert result.log_concavity_rows[0].neighbor_product == 6
    assert not result.log_concavity_rows[0].holds


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
    source = FiniteIntegerSequence(values=(1,) * 10_000)
    result = sequence_order_shape(source)
    assert result.weak_unimodal_peak_positions == tuple(range(10_000))
