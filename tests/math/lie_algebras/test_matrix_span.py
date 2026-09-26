from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lie_algebras.matrix_span._models import (
    LieMatrixSpanRealization,
    LieMatrixSpanRequest,
)
from jacobian.math.lie_algebras.matrix_span.operations import (
    lie_algebra_from_matrix_span,
)


def _matrix(entries: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "entries": [[{"num": value, "den": 1} for value in row] for row in entries],
    }


def _request(*matrices: tuple[tuple[int, ...], ...]) -> LieMatrixSpanRequest:
    return LieMatrixSpanRequest.model_validate(
        {"matrices": [_matrix(matrix) for matrix in matrices]}
    )


def _oracle_bracket(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[Fraction, ...]:
    """Independent exact matrix multiplication oracle used by these fixtures."""
    n = len(left)
    return tuple(
        sum(
            (
                Fraction(left[i][k] * right[k][j] - right[i][k] * left[k][j])
                for k in range(n)
            ),
            Fraction(),
        )
        for i in range(n)
        for j in range(n)
    )


E = ((0, 1), (0, 0))
F = ((0, 0), (1, 0))
H = ((1, 0), (0, -1))


def test_sl2_matrix_span_constructs_induced_bracket_and_roundtrips() -> None:
    request = _request(E, F, H)
    result = lie_algebra_from_matrix_span(request)
    assert result.algebra.basis == ("M0", "M1", "M2")
    assert [
        (item.i, item.j, item.k, item.coefficient.as_fraction())
        for item in result.algebra.structure_constants
    ] == [(0, 1, 2, Fraction(1)), (0, 2, 0, Fraction(-2)), (1, 2, 1, Fraction(2))]
    for item in result.algebra.structure_constants:
        oracle = _oracle_bracket((E, F, H)[item.i], (E, F, H)[item.j])
        expected = tuple(
            Fraction(value) for value in (E, F, H)[item.k][0] + (E, F, H)[item.k][1]
        )
        assert oracle == tuple(
            item.coefficient.as_fraction() * value for value in expected
        )
    assert LieMatrixSpanRealization.model_validate(result.model_dump()) == result


def test_nonclosed_or_dependent_span_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="not closed"):
        lie_algebra_from_matrix_span(_request(E, F))
    with pytest.raises(OperationDomainValidationError, match="independent"):
        lie_algebra_from_matrix_span(_request(E, E))


def test_one_dimensional_scalar_span_has_zero_bracket() -> None:
    identity = ((1, 0), (0, 1))
    result = lie_algebra_from_matrix_span(_request(identity))
    assert result.algebra.structure_constants == ()
    assert result.matrix_basis == _request(identity).matrices


def test_raw_scalar_height_is_rejected_before_matrix_canonicalization() -> None:
    payload = {"matrices": [_matrix(((10**64, 0), (0, 0)))]}
    with pytest.raises(ValidationError, match="limited to 64"):
        LieMatrixSpanRequest.model_validate(payload)
