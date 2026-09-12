"""Exact aperiodic and cyclic autocorrelation contracts."""

from collections.abc import Callable
from typing import cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
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
from jacobian.math.number_theory.sequences.core.values import IntegerSequence


def values(result: AutocorrelationResult) -> list[tuple[int, int]]:
    return [(cell.lag, cell.value) for cell in result.cells]


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
        '{"values":["1",{"num":"3","den":"1"},2]}'
    )
    assert source.values == (
        CanonicalRational(num=1, den=1),
        CanonicalRational(num=3, den=1),
        CanonicalRational(num=2, den=1),
    )


def test_rational_profile_reports_first_witnesses_and_vacuous_edges() -> None:
    result = sequence_order_shape(
        FiniteRationalSequence(
            values=tuple(
                CanonicalRational(num=value, den=1) for value in (3, 1, 2, 0, -1)
            )
        )
    )
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
            cast(FiniteRationalSequence, IntegerSequence(values=(1, 2)))
        )


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


def test_order_shape_rejects_complete_profile_output_explosion() -> None:
    source = FiniteIntegerSequence(values=(1,) * 100_000)
    with pytest.raises(
        OperationResourceAdmissionError,
        match=str(MAX_ORDER_SHAPE_RESULT_ALLOCATIONS),
    ):
        sequence_order_shape(source)
