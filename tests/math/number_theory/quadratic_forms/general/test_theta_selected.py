"""Exact representation numbers and fibers of positive-definite forms."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational as Q
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_THETA_SELECTED_INDICES,
    ThetaRepresentingVectorsRequest,
    ThetaSelectedCoefficientsRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import TOOLS
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_representing_vectors,
    theta_selected_coefficients,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(
    diagonal: tuple[int, ...], cross: tuple[tuple[int, int, int], ...] = ()
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=tuple("xyz"[: len(diagonal)]),
        diagonal_coefficients=tuple(Q.from_integer_ratio(d, 1) for d in diagonal),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left, right=right, coefficient=Q.from_integer_ratio(c, 1)
            )
            for left, right, c in cross
        ),
    )


def _brute_force(
    form: RationalQuadraticForm, radius: int, indices: tuple[int, ...]
) -> tuple[dict[int, int], dict[int, list[tuple[int, ...]]]]:
    """Independent oracle: enumerate a large box directly and count.

    Deliberately shares nothing with the adjugate bound in the kernel, so it
    checks the bound rather than re-deriving it.
    """

    diagonal = tuple(c.num for c in form.diagonal_coefficients)
    crosses = tuple((t.left, t.right, t.coefficient.num) for t in form.cross_terms)
    wanted = set(indices)
    counts = dict.fromkeys(indices, 0)
    fibers: dict[int, list[tuple[int, ...]]] = {index: [] for index in indices}
    ranges = [range(-radius, radius + 1)] * len(diagonal)
    for vector in product(*ranges):
        value = sum(a * x * x for a, x in zip(diagonal, vector, strict=True))
        value += sum(c * vector[left] * vector[right] for left, right, c in crosses)
        if value in wanted:
            counts[value] += 1
            fibers[value].append(vector)
    return counts, {k: sorted(v) for k, v in fibers.items()}


HEXAGONAL = _form((1, 1), ((0, 1, 1),))
UNARY = _form((1,))


def test_selected_coefficients_match_an_independent_box_enumeration() -> None:
    indices = (0, 3, 4, 7, 12)
    result = theta_selected_coefficients(HEXAGONAL, indices)
    expected, _ = _brute_force(HEXAGONAL, radius=12, indices=indices)
    assert tuple(row.coefficient for row in result.coefficients) == tuple(
        expected[index] for index in indices
    )
    # the requested index axis, not a positional one
    assert tuple(row.index for row in result.coefficients) == indices


def test_representation_fibers_match_an_independent_box_enumeration() -> None:
    indices = (0, 3, 4)
    result = theta_representing_vectors(HEXAGONAL, indices)
    _, expected = _brute_force(HEXAGONAL, radius=12, indices=indices)
    for row in result.rows:
        assert [tuple(c.num for c in v.coordinates) for v in row.vectors] == (
            expected[row.index]
        )


def test_a_non_positive_definite_form_is_refused_not_truncated() -> None:
    """The unsound case from the withdrawn surface: a truncated table."""
    indefinite = _form((1, 1), ((0, 1, -3),))
    for call in (
        lambda: theta_selected_coefficients(indefinite, (1,)),
        lambda: theta_representing_vectors(indefinite, (1,)),
    ):
        with pytest.raises(OperationDomainValidationError) as error:
            call()
        assert (
            error.value.errors()[0]["type"]
            == "quadratic_form.theta_requires_positive_definite"
        )


def test_degenerate_form_is_refused() -> None:
    # A zero diagonal coefficient makes the polar matrix singular, which is
    # degenerate rather than positive-definite. (Zero *cross* terms are not
    # representable in the carrier at all.)
    degenerate = _form((1, 0))
    with pytest.raises(OperationDomainValidationError) as error:
        theta_selected_coefficients(degenerate, (1,))
    assert (
        error.value.errors()[0]["type"]
        == "quadratic_form.theta_requires_positive_definite"
    )


def test_unary_representation_counts_are_the_sum_of_two_squares_ones() -> None:
    """r_{x^2}(n) is 2 when n is a nonzero square and 1 at zero, else 0."""
    indices = (0, 1, 2, 3, 4, 9)
    result = theta_selected_coefficients(UNARY, indices)
    assert [row.coefficient for row in result.coefficients] == [1, 2, 0, 0, 2, 2]


def test_empty_fiber_is_reported_as_zero_not_omitted() -> None:
    result = theta_selected_coefficients(UNARY, (1, 2))
    assert [row.coefficient for row in result.coefficients] == [2, 0]
    fibers = theta_representing_vectors(UNARY, (1, 2))
    assert [len(row.vectors) for row in fibers.rows] == [2, 0]


def test_request_requires_strictly_increasing_indices() -> None:
    with pytest.raises(ValidationError) as decreasing:
        ThetaSelectedCoefficientsRequest(form=UNARY, indices=(3, 1))
    assert decreasing.value.errors()[0]["type"] == (
        "quadratic_form.theta_indices_not_increasing"
    )
    with pytest.raises(ValidationError):
        ThetaRepresentingVectorsRequest(form=UNARY, indices=(1, 1))
    with pytest.raises(ValidationError):
        ThetaSelectedCoefficientsRequest(form=UNARY, indices=())


def test_too_many_selected_indices_are_refused() -> None:
    with pytest.raises(ValidationError):
        ThetaSelectedCoefficientsRequest(
            form=UNARY, indices=tuple(range(MAX_THETA_SELECTED_INDICES + 1))
        )


def test_sparse_high_index_is_admitted_without_charging_the_dense_prefix() -> None:
    """A selected-index request retains only the selected rows.

    It reused the dense-prefix output estimate `(cutoff + 1) * (count_digits + 1)`,
    so a single index at 20000 charged for 20,001 coefficients that are computed
    but never returned, and the request was refused even though the box holds
    only 401 vectors and one coefficient is retained.
    """
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
        MAX_THETA_PREFIX_OUTPUT_DIGITS,
    )
    from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
        _admit_box_and_output,
    )

    square = _form((1,))

    # one index high enough that the dense charge would exceed the envelope
    dense_charge = 1 + (20000 + 1) * (3 + 1)
    assert dense_charge > MAX_THETA_PREFIX_OUTPUT_DIGITS

    result = theta_selected_coefficients(square, (20000,))

    assert tuple(row.index for row in result.coefficients) == (20000,)
    # r(x^2 = 20000) is 0; the search is still complete, not truncated.
    assert result.coefficients[0].coefficient == 0

    # the dense-prefix operation still charges the whole prefix
    with pytest.raises(OperationResourceAdmissionError):
        _admit_box_and_output(
            square,
            20000,
            1,
            0,
            0,
            1,
            (1,),
        )


def test_an_empty_index_selection_is_refused_rather_than_raising_index_error() -> None:
    """The cutoff is read positionally, so the native boundary owns the rule.

    The request schema already requires a non-empty selection, but a native
    caller bypasses it. Without the boundary check an empty tuple escaped as a
    bare ``IndexError`` from ``indices[-1]``, which is neither this operation's
    documented failure nor a classified one.
    """
    with pytest.raises(OperationDomainValidationError) as refusal:
        theta_selected_coefficients(UNARY, ())
    assert refusal.value.errors()[0]["type"] == "quadratic_form.theta_indices"


def test_a_non_increasing_index_selection_is_refused() -> None:
    with pytest.raises(OperationDomainValidationError):
        theta_selected_coefficients(UNARY, (2, 1))
    with pytest.raises(OperationDomainValidationError):
        theta_selected_coefficients(UNARY, (1, 1))


@pytest.mark.parametrize(
    "operation", [theta_selected_coefficients, theta_representing_vectors]
)
@pytest.mark.parametrize(
    "indices",
    [
        None,
        [],
        (),
        (-1,),
        (1_000_000_001,),
        (True,),
        (0.5,),
        ("1",),
        ([0],),
        (2, 1),
        (1, 1),
        tuple(range(129)),
    ],
)
def test_selected_theta_operations_reject_invalid_native_indices(
    operation: Callable[[RationalQuadraticForm, tuple[int, ...]], object],
    indices: object,
) -> None:
    with pytest.raises(OperationDomainValidationError) as refusal:
        operation(UNARY, cast(tuple[int, ...], indices))
    assert refusal.value.errors()[0]["type"] == "quadratic_form.theta_indices"


def test_selected_theta_at_index_count_boundary_round_trips() -> None:
    indices = tuple(range(MAX_THETA_SELECTED_INDICES))
    expected_counts, expected_fibers = _brute_force(UNARY, radius=12, indices=indices)

    counts = theta_selected_coefficients(UNARY, indices)
    counts = type(counts).model_validate_json(counts.model_dump_json())
    assert counts.form == UNARY
    assert {
        row.index: row.coefficient for row in counts.coefficients
    } == expected_counts

    fibers = theta_representing_vectors(UNARY, indices)
    fibers = type(fibers).model_validate_json(fibers.model_dump_json())
    assert fibers.form == UNARY
    assert {
        row.index: [
            tuple(coordinate.num for coordinate in vector.coordinates)
            for vector in row.vectors
        ]
        for row in fibers.rows
    } == expected_fibers


def test_the_theta_pair_publishes_its_integral_precondition() -> None:
    """Both theta operations enforce integrality; discovery must say so.

    The kernel refuses a non-integral form because the polar matrix it applies
    Sylvester's criterion to has integral entries. Both descriptions previously
    advertised a "rational form", so a caller could only discover the rule by
    being refused.
    """
    theta_ids = (
        "quadratic_form.theta_selected_coefficients.compute",
        "quadratic_form.representing_vectors.compute",
    )
    for operation_id in theta_ids:
        tool = next(t for t in TOOLS if t.operation_id == operation_id)
        assert "integral polynomial coefficients" in tool.description
        assert "non-integral form is refused" in tool.description
