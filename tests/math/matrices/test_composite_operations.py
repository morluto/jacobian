"""Domain tests for the matrix permanent, Kronecker product, and partial trace."""

from __future__ import annotations

import pytest
from tests.support.rationals import rational_payload as q

from jacobian._exact import CanonicalRational
from jacobian.math.matrices._operation_models import (
    MatrixKroneckerProductRequest,
    MatrixKroneckerProductResult,
    MatrixPartialTraceRequest,
    MatrixPartialTraceResult,
    MatrixPermanentResult,
    SquareRationalMatrixRequest,
)
from jacobian.math.matrices._operations import (
    compute_kronecker_product,
    compute_partial_trace,
    compute_permanent,
)
from jacobian.math.matrices.values import RationalMatrix


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(num, den)


def _matrix(entries: list[list[dict[str, str]]]) -> RationalMatrix:
    return RationalMatrix(entries=tuple(tuple(row) for row in entries))


def _sq_request(entries: list[list[dict[str, str]]]) -> SquareRationalMatrixRequest:
    return SquareRationalMatrixRequest(matrix=_matrix(entries))


def test_matrix_permanent_of_two_by_two() -> None:
    request = _sq_request([[q(1), q(2)], [q(3), q(4)]])
    result = compute_permanent(request)
    assert isinstance(result, MatrixPermanentResult)
    assert result.permanent == _cr(10)


def test_matrix_permanent_of_identity() -> None:
    request = _sq_request([[q(1), q(0)], [q(0), q(1)]])
    assert compute_permanent(request).permanent == _cr(1)


def test_matrix_permanent_of_all_ones_two_by_two() -> None:
    request = _sq_request([[q(1), q(1)], [q(1), q(1)]])
    assert compute_permanent(request).permanent == _cr(2)


def test_matrix_permanent_of_three_by_three_all_ones() -> None:
    request = _sq_request([[q(1), q(1), q(1)], [q(1), q(1), q(1)], [q(1), q(1), q(1)]])
    assert compute_permanent(request).permanent == _cr(6)


def test_matrix_permanent_of_rationals() -> None:
    request = _sq_request([[q(1, 2), q(1)], [q(1), q(1, 2)]])
    assert compute_permanent(request).permanent == _cr(5, 4)


def test_matrix_permanent_requires_square() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="square"):
        SquareRationalMatrixRequest.model_validate(
            {"matrix": {"entries": [[q(1), q(2)]]}}
        )


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
    with pytest.raises(ValueError, match="composite matrix"):
        MatrixPartialTraceRequest.model_validate(
            {
                "matrix": {"entries": [[q(1), q(0)], [q(0), q(1)]]},
                "traced_dimension": 2,
                "kept_dimension": 2,
            }
        )


def test_partial_trace_rejects_non_square_composite() -> None:
    with pytest.raises(ValueError, match="composite matrix"):
        MatrixPartialTraceRequest.model_validate(
            {
                "matrix": {"entries": [[q(1), q(0), q(0), q(0)]]},
                "traced_dimension": 2,
                "kept_dimension": 2,
            }
        )
