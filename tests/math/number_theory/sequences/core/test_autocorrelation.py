"""Exact aperiodic and cyclic autocorrelation contracts."""

from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationRequest,
    AutocorrelationResult,
)
from jacobian.math.number_theory.sequences.core.operations import (
    aperiodic_autocorrelation,
    cyclic_autocorrelation,
    sequence_order_shape,
)


def values(result: AutocorrelationResult) -> list[tuple[int, int]]:
    return [(cell.lag, cell.value) for cell in result.cells]


def test_aperiodic_uses_signed_nonwrapping_lags() -> None:
    result = aperiodic_autocorrelation(AutocorrelationRequest(values=(1, 2, 3)))
    assert values(result) == [(-2, 3), (-1, 8), (0, 14), (1, 8), (2, 3)]
    assert result.source.values == (1, 2, 3)


def test_cyclic_uses_one_complete_residue_axis() -> None:
    result = cyclic_autocorrelation(AutocorrelationRequest(values=(1, 2, 3)))
    assert values(result) == [(0, 14), (1, 11), (2, 11)]


def test_empty_sequence_has_empty_profiles() -> None:
    source = AutocorrelationRequest(values=())
    assert aperiodic_autocorrelation(source).cells == ()
    assert cyclic_autocorrelation(source).cells == ()


def test_order_shape_retains_all_flat_peak_positions_and_signed_rows() -> None:
    source = AutocorrelationRequest(values=(1, 3, 3, 2))
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
    result = sequence_order_shape(AutocorrelationRequest(values=(-2, 0, -3)))
    assert not result.is_nonnegative
    assert result.has_internal_zero
    assert result.log_concavity_rows[0].neighbor_product == 6
    assert not result.log_concavity_rows[0].holds
