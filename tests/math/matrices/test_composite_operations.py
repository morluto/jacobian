"""Domain tests for the matrix permanent, Kronecker product, and partial trace."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices._operation_models import (
    MatrixKroneckerProductRequest,
    MatrixKroneckerProductResult,
    MatrixPartialTraceRequest,
    MatrixPartialTraceResult,
    MatrixPermanentRequest,
    MatrixPermanentResult,
)
from jacobian.math.matrices._tools import (
    compute_kronecker_product,
    compute_partial_trace,
    compute_permanent,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
    RationalMatrix,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


def q(num: int, den: int = 1) -> CanonicalRational:
    return _cr(num, den)


def _identity_entries(size: int) -> tuple[tuple[CanonicalRational, ...], ...]:
    return tuple(
        tuple(_cr(1) if index == column else _cr(0) for column in range(size))
        for index in range(size)
    )


def _matrix(entries: list[list[CanonicalRational]]) -> RationalMatrix:
    return RationalMatrix(
        entries=tuple(tuple(value for value in row) for row in entries)
    )


def _permanent_request(
    entries: list[list[CanonicalRational]],
) -> MatrixPermanentRequest:
    return MatrixPermanentRequest(matrix=_matrix(entries))


def test_matrix_permanent_of_two_by_two() -> None:
    request = _permanent_request([[q(1), q(2)], [q(3), q(4)]])
    result = compute_permanent(request)
    assert isinstance(result, MatrixPermanentResult)
    assert result.permanent == _cr(10)


def test_matrix_permanent_of_identity() -> None:
    request = _permanent_request([[q(1), q(0)], [q(0), q(1)]])
    assert compute_permanent(request).permanent == _cr(1)


def test_matrix_permanent_of_all_ones_two_by_two() -> None:
    request = _permanent_request([[q(1), q(1)], [q(1), q(1)]])
    assert compute_permanent(request).permanent == _cr(2)


def test_matrix_permanent_of_three_by_three_all_ones() -> None:
    request = _permanent_request(
        [[q(1), q(1), q(1)], [q(1), q(1), q(1)], [q(1), q(1), q(1)]]
    )
    assert compute_permanent(request).permanent == _cr(6)


def test_matrix_permanent_of_rationals() -> None:
    request = _permanent_request([[q(1, 2), q(1)], [q(1), q(1, 2)]])
    assert compute_permanent(request).permanent == _cr(5, 4)


def test_matrix_permanent_requires_square() -> None:
    request = MatrixPermanentRequest.model_validate(
        {"matrix": {"entries": [[q(1), q(2)]]}}
    )
    with pytest.raises(OperationDomainValidationError):
        compute_permanent(request)


def test_permanent_request_rejects_order_above_the_ryser_work_budget() -> None:
    from pydantic import ValidationError

    from jacobian.math.matrices._operation_models import MAX_PERMANENT_MATRIX_ORDER

    oversized = RationalMatrix(
        entries=_identity_entries(MAX_PERMANENT_MATRIX_ORDER + 1)
    )
    with pytest.raises(
        ValidationError,
        match=f"limited to {MAX_PERMANENT_MATRIX_ORDER} rows and columns",
    ):
        MatrixPermanentRequest(matrix=oversized)


def test_permanent_admits_fourteen_by_fourteen_all_ones() -> None:
    """Ryser envelope reaches order 14: 14! via the owner-local Gray-code kernel."""

    from jacobian.math.matrices._operation_models import MAX_PERMANENT_MATRIX_ORDER

    assert MAX_PERMANENT_MATRIX_ORDER == 14
    entries = [[q(1) for _ in range(14)] for _ in range(14)]
    result = compute_permanent(_permanent_request(entries))
    assert result.permanent == _cr(87178291200)


def test_gray_code_ryser_matches_direct_definition() -> None:
    """The Gray-code kernel agrees with the Leibniz definition on rationals."""

    import itertools
    from fractions import Fraction

    from jacobian.math.matrices.operations import _permanent_of_fractions

    entries = (
        (Fraction(1, 2), Fraction(-1, 3), Fraction(2)),
        (Fraction(3), Fraction(1, 4), Fraction(-2, 5)),
        (Fraction(-1, 6), Fraction(5), Fraction(1, 7)),
    )
    expected = sum(
        entries[0][p[0]] * entries[1][p[1]] * entries[2][p[2]]
        for p in itertools.permutations(range(3))
    )
    assert _permanent_of_fractions(entries) == expected


def test_gray_code_ryser_fallback_handles_oversized_cleared_entries() -> None:
    """Denominator-clearing overflow uses the entrywise Fraction accumulation."""

    from fractions import Fraction

    from jacobian.math.matrices.operations import (
        MAX_RYSER_CLEARED_ENTRY_DIGITS,
        _permanent_of_fractions,
        _ryser_instrument_digits,
    )

    entries = tuple(
        tuple(Fraction(10**2000, 3) for _ in range(3)) for _ in range(3)
    )
    assert _ryser_instrument_digits(entries) > MAX_RYSER_CLEARED_ENTRY_DIGITS
    assert _permanent_of_fractions(entries) == Fraction(2 * 10**6000, 9)


def test_permanent_request_admits_the_ryser_work_boundary() -> None:
    from jacobian.math.matrices._operation_models import MAX_PERMANENT_MATRIX_ORDER

    boundary = RationalMatrix(entries=_identity_entries(MAX_PERMANENT_MATRIX_ORDER))
    assert MatrixPermanentRequest(matrix=boundary).matrix == boundary


def test_native_permanent_applies_the_ryser_order_admission() -> None:
    """The native SymPy entry point must share the typed exponential bound."""

    import sympy

    from jacobian.math.matrices.operations import permanent

    oversized = sympy.eye(15)
    with pytest.raises(ValueError, match="Ryser work budget"):
        permanent(oversized)


def test_kronecker_request_rejects_operands_above_the_computation_dimension() -> None:
    from pydantic import ValidationError

    tall = RationalMatrix(
        entries=tuple((_cr(1),) for _ in range(MAX_MATRIX_DIMENSION + 1))
    )
    unit = RationalMatrix(entries=((_cr(1),),))
    with pytest.raises(ValidationError):
        MatrixKroneckerProductRequest(left=tall, right=unit)
    with pytest.raises(ValidationError):
        MatrixKroneckerProductRequest(left=unit, right=tall)


def test_kronecker_request_rejects_products_beyond_the_operation_axis_budget() -> None:
    from jacobian.math.matrices._operation_models import (
        MAX_KRONECKER_PRODUCT_AXIS,
    )

    factor = RationalMatrix(entries=_identity_entries(8))
    request = MatrixKroneckerProductRequest(left=factor, right=factor)
    with pytest.raises(OperationDomainValidationError) as excinfo:
        compute_kronecker_product(request)
    assert excinfo.value.errors()[0]["type"] == "matrix.budget_exceeded"
    assert MAX_KRONECKER_PRODUCT_AXIS < 8 * 8 <= MAX_RATIONAL_MATRIX_ORDER


def test_kronecker_product_admits_the_operation_axis_budget_boundary() -> None:
    from jacobian.math.matrices._operation_models import (
        MAX_KRONECKER_PRODUCT_AXIS,
    )

    left = RationalMatrix(entries=_identity_entries(5))
    right = RationalMatrix(entries=_identity_entries(10))
    result = compute_kronecker_product(
        MatrixKroneckerProductRequest(left=left, right=right)
    )
    assert len(result.product.entries) == MAX_KRONECKER_PRODUCT_AXIS
    assert len(result.product.entries[0]) == MAX_KRONECKER_PRODUCT_AXIS


def test_kronecker_product_within_the_operation_axis_budget() -> None:
    side = 7
    request = MatrixKroneckerProductRequest(
        left=RationalMatrix(entries=_identity_entries(side)),
        right=RationalMatrix(entries=_identity_entries(side)),
    )
    result = compute_kronecker_product(request)
    order = side * side
    assert order < MAX_RATIONAL_MATRIX_ORDER
    assert result.left_rows == result.right_rows == side
    assert len(result.product.entries) == order
    assert result.product.entries[0][0] == _cr(1)


def test_kronecker_product_of_two_by_two() -> None:
    request = MatrixKroneckerProductRequest.model_validate(
        {
            "left": {"entries": [[q(1), q(2)], [q(3), q(4)]]},
            "right": {"entries": [[q(0), q(5)], [q(6), q(7)]]},
        }
    )
    result = compute_kronecker_product(request)
    assert isinstance(result, MatrixKroneckerProductResult)
    assert result.left_rows == 2
    assert result.left_columns == 2
    assert result.right_rows == 2
    assert result.right_columns == 2
    expected = [
        [_cr(0), _cr(5), _cr(0), _cr(10)],
        [_cr(6), _cr(7), _cr(12), _cr(14)],
        [_cr(0), _cr(15), _cr(0), _cr(20)],
        [_cr(18), _cr(21), _cr(24), _cr(28)],
    ]
    assert result.product.entries == tuple(tuple(row) for row in expected)


def test_kronecker_product_with_identity() -> None:
    request = MatrixKroneckerProductRequest.model_validate(
        {
            "left": {"entries": [[q(1), q(0)], [q(0), q(1)]]},
            "right": {"entries": [[q(5), q(6)], [q(7), q(8)]]},
        }
    )
    result = compute_kronecker_product(request)
    expected = [
        [_cr(5), _cr(6), _cr(0), _cr(0)],
        [_cr(7), _cr(8), _cr(0), _cr(0)],
        [_cr(0), _cr(0), _cr(5), _cr(6)],
        [_cr(0), _cr(0), _cr(7), _cr(8)],
    ]
    assert result.product.entries == tuple(tuple(row) for row in expected)


def test_partial_trace_of_diagonal_kronecker_product() -> None:
    # A = diag(1, 2), B = eye(2); A (x) B is diag(1, 1, 2, 2)
    composite = [
        [q(1), q(0), q(0), q(0)],
        [q(0), q(1), q(0), q(0)],
        [q(0), q(0), q(2), q(0)],
        [q(0), q(0), q(0), q(2)],
    ]
    request = MatrixPartialTraceRequest.model_validate(
        {
            "matrix": {"entries": composite},
            "traced_dimension": 2,
            "kept_dimension": 2,
        }
    )
    result = compute_partial_trace(request)
    assert isinstance(result, MatrixPartialTraceResult)
    # trace(A) * B = (1+2) * I = 3*I
    expected = [[_cr(3), _cr(0)], [_cr(0), _cr(3)]]
    assert result.reduced_matrix.entries == tuple(tuple(row) for row in expected)


def test_partial_trace_of_full_two_by_two_factors() -> None:
    # A = [[1,2],[3,4]], B = [[0,5],[6,7]]
    # A (x) B is the Kronecker product; partial trace over A gives trace(A)*B
    import sympy

    from jacobian.math.matrices import kronecker_product

    a = sympy.Matrix([[1, 2], [3, 4]])
    b = sympy.Matrix([[0, 5], [6, 7]])
    kron = kronecker_product(a, b)
    composite = []
    for i in range(kron.rows):
        composite.append(
            [q(int(kron[i, j].p), int(kron[i, j].q)) for j in range(kron.cols)]
        )
    request = MatrixPartialTraceRequest.model_validate(
        {
            "matrix": {"entries": composite},
            "traced_dimension": 2,
            "kept_dimension": 2,
        }
    )
    result = compute_partial_trace(request)
    # trace(A) = 5, so reduced = 5 * B
    trace_a = 5
    expected = [
        [_cr(0 * trace_a), _cr(5 * trace_a)],
        [_cr(6 * trace_a), _cr(7 * trace_a)],
    ]
    assert result.reduced_matrix.entries == tuple(tuple(row) for row in expected)


def test_partial_trace_rejects_non_composite_shape() -> None:
    request = MatrixPartialTraceRequest.model_validate(
        {
            "matrix": {"entries": [[q(1), q(0)], [q(0), q(1)]]},
            "traced_dimension": 2,
            "kept_dimension": 2,
        }
    )
    with pytest.raises(OperationDomainValidationError):
        compute_partial_trace(request)


def test_partial_trace_rejects_non_square_composite() -> None:
    request = MatrixPartialTraceRequest.model_validate(
        {
            "matrix": {"entries": [[q(1), q(0), q(0), q(0)]]},
            "traced_dimension": 2,
            "kept_dimension": 2,
        }
    )
    with pytest.raises(OperationDomainValidationError):
        compute_partial_trace(request)
