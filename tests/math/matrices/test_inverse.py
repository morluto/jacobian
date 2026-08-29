"""Public matrix-inverse operation tests.

These tests invoke the matrices-owned declaration rather than importing the
private Smith-normal-form kernel.  The kernel is exercised through the
``matrix.inverse.compute`` contract, while this module asserts the observable
rational inverse and request-boundary behavior.
"""

from __future__ import annotations

from fractions import Fraction
from random import Random
from typing import Any

import pytest

from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.matrices._operation_models import MatrixInverseResult
from jacobian.math.matrices._tools import TOOLS
from jacobian.math.matrices.operations import inverse_result
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
)


def _operation(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _run_inverse(entries: list[list[str]]) -> MatrixInverseResult:
    operation = _operation("matrix.inverse.compute")
    request = operation.request_type.model_validate({"matrix": {"entries": entries}})
    result = operation.run(request)
    assert isinstance(result, MatrixInverseResult)
    return result


def _random_unimodular(random: Random, size: int) -> list[list[int]]:
    matrix = [[int(row == column) for column in range(size)] for row in range(size)]
    for _ in range(size * 6):
        i, j = random.randrange(size), random.randrange(size)
        if i == j:
            if random.random() < 0.5:
                matrix[i] = [-value for value in matrix[i]]
            continue
        factor = random.randint(-5, 5)
        matrix[i] = [
            value + factor * other
            for value, other in zip(matrix[i], matrix[j], strict=True)
        ]
    if random.random() < 0.5:
        a, b = random.randrange(size), random.randrange(size)
        matrix[a], matrix[b] = matrix[b], matrix[a]
    return matrix


def _multiply(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (
                    left[row][index] * right[index][column]
                    for index in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _fraction_entries(entries: list[list[str]]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(int(value)) for value in row) for row in entries)


def _result_entries(result: MatrixInverseResult) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row) for row in result.inverse.entries
    )


def test_inverse_operation_returns_exact_two_sided_inverse() -> None:
    source = [["1", "1"], ["0", "1"]]
    result = _run_inverse(source)
    inverse = _result_entries(result)

    assert inverse == ((Fraction(1), Fraction(-1)), (Fraction(0), Fraction(1)))
    assert _multiply(_fraction_entries(source), inverse) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def test_inverse_operation_accepts_determinant_minus_one() -> None:
    source = [["0", "1"], ["1", "0"]]
    result = _run_inverse(source)
    inverse = _result_entries(result)

    assert inverse == ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
    assert _multiply(_fraction_entries(source), inverse) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def test_inverse_operation_returns_rational_values_for_non_unimodular_matrix() -> None:
    result = _run_inverse([["2", "0"], ["0", "1"]])

    assert _result_entries(result) == (
        (Fraction(1, 2), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_inverse_operation_round_trips_random_unimodular_matrices(size: int) -> None:
    random = Random(1127 + size)
    for _ in range(20):
        source = _random_unimodular(random, size)
        source_wire = [[str(value) for value in row] for row in source]
        inverse = _result_entries(_run_inverse(source_wire))
        source_fraction = tuple(
            tuple(Fraction(value) for value in row) for row in source
        )
        identity = tuple(
            tuple(Fraction(int(row == column)) for column in range(size))
            for row in range(size)
        )
        assert _multiply(source_fraction, inverse) == identity
        assert _multiply(inverse, source_fraction) == identity


def test_inverse_result_accepts_order_33_integer_matrix() -> None:
    """The inverse operation owns the widened integer-matrix envelope."""

    matrix = IntegerMatrix(
        entries=tuple(
            tuple(
                "1" if row == column else "0"
                for column in range(MAX_MATRIX_DIMENSION + 1)
            )
            for row in range(MAX_MATRIX_DIMENSION + 1)
        )
    )
    result = inverse_result(matrix)
    assert _result_entries(result) == tuple(
        tuple(Fraction(int(row == column)) for column in range(33)) for row in range(33)
    )


def test_inverse_operation_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError):
        _run_inverse([])


def test_inverse_operation_rejects_non_square_matrices() -> None:
    with pytest.raises(ValueError):
        _run_inverse([["1", "0"]])


def test_inverse_operation_rejects_singular_matrices() -> None:
    with pytest.raises(ValueError):
        _run_inverse([["1", "2"], ["2", "4"]])


def test_flint_inverse_exceeds_shared_integer_matrix_order() -> None:
    order = 80
    source = [
        [str(int(row == column or column == row + 1)) for column in range(order)]
        for row in range(order)
    ]

    inverse = _result_entries(_run_inverse(source))

    expected = tuple(
        tuple(
            Fraction((-1) ** (column - row)) if column >= row else Fraction(0)
            for column in range(order)
        )
        for row in range(order)
    )
    assert inverse == expected
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(order))
        for row in range(order)
    )
    assert _multiply(_fraction_entries(source), inverse) == identity
    assert _multiply(inverse, _fraction_entries(source)) == identity


def _identity_entries(order: int) -> list[list[str]]:
    return [
        [str(int(row == column)) for column in range(order)] for row in range(order)
    ]


def test_inverse_admits_sparse_identity_with_tiny_output() -> None:
    order = 100
    source = _identity_entries(order)
    inverse = _result_entries(_run_inverse(source))
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(order))
        for row in range(order)
    )

    assert inverse == identity
    assert _multiply(_fraction_entries(source), inverse) == identity
    assert _multiply(inverse, _fraction_entries(source)) == identity


def test_inverse_admits_diagonal_max_height_output() -> None:
    order = 100
    diagonal = "1" + "0" * 255
    source = [
        [diagonal if row == column else "0" for column in range(order)]
        for row in range(order)
    ]

    inverse = _result_entries(_run_inverse(source))
    expected = tuple(
        tuple(
            Fraction(1, 10**255) if row == column else Fraction(0)
            for column in range(order)
        )
        for row in range(order)
    )
    assert inverse == expected


def test_inverse_admits_bounded_rank_one_update() -> None:
    order = 100
    height = 10**255
    vector = tuple(height if index % 2 == 0 else -height for index in range(order))
    source = [
        [str((1 if row == column else 0) + vector[column]) for column in range(order)]
        for row in range(order)
    ]

    inverse = _result_entries(_run_inverse(source))
    expected = tuple(
        tuple(
            Fraction((1 if row == column else 0) - vector[column])
            for column in range(order)
        )
        for row in range(order)
    )

    assert inverse == expected


def test_inverse_rejects_dense_output_work_before_backend() -> None:
    order = 128
    tall = "1" + "0" * 99
    source = [[tall] * order for _ in range(order)]

    with pytest.raises(ValueError, match="exact output budget"):
        _run_inverse(source)


def _entry_axis_limit(schema: dict[str, Any], field: str) -> int:
    field_schema = schema["properties"][field]
    entries = field_schema.get("properties", {}).get("entries")
    if entries is None:
        reference = field_schema["$ref"]
        name = reference.rsplit("/", 1)[-1]
        entries = schema["$defs"][name]["properties"]["entries"]
    return int(entries["maxItems"])


def test_inverse_reuses_canonical_integer_matrix() -> None:
    from jacobian.math.matrices._operation_models import (
        NonsingularIntegerMatrixRequest,
    )
    from jacobian.math.matrices._tools import compute_inverse
    from jacobian.math.matrices.values import IntegerMatrix

    matrix = IntegerMatrix.model_validate({"entries": [["1", "0"], ["0", "1"]]})
    request = NonsingularIntegerMatrixRequest(matrix=matrix)

    assert request.matrix is matrix
    inverse = _result_entries(compute_inverse(request))
    assert inverse == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))


def test_non_inverse_integer_requests_keep_order_32_envelope() -> None:
    from pydantic import ValidationError

    from jacobian.math.lattices._hnf import compute_hermite_normal_form
    from jacobian.math.lattices._lattice import reduce_lattice_basis
    from jacobian.math.lattices._models import (
        HermiteNormalFormRequest,
        LatticeReductionRequest,
    )
    from jacobian.math.matrices._operation_models import (
        IntegerMatrixRequest,
        NonsingularIntegerMatrixRequest,
        SquareIntegerMatrixRequest,
    )
    from jacobian.math.matrices.values import IntegerMatrix

    order = 33
    entries = _identity_entries(order)
    matrix = IntegerMatrix.model_validate({"entries": entries})
    inverse_request = NonsingularIntegerMatrixRequest(matrix=matrix)

    assert len(matrix.entries) == order
    assert inverse_request.matrix is matrix

    assert IntegerMatrixRequest.model_validate({"matrix": {"entries": entries}})
    assert IntegerMatrixRequest(matrix=matrix).matrix is matrix
    with pytest.raises(ValidationError):
        SquareIntegerMatrixRequest.model_validate({"matrix": {"entries": entries}})
    with pytest.raises(ValidationError):
        SquareIntegerMatrixRequest(matrix=matrix)
    with pytest.raises(ValidationError):
        LatticeReductionRequest.model_validate({"basis": {"entries": entries}})
    with pytest.raises(ValidationError):
        LatticeReductionRequest(basis=matrix)
    with pytest.raises(ValidationError):
        HermiteNormalFormRequest.model_validate({"matrix": {"entries": entries}})
    with pytest.raises(ValidationError):
        HermiteNormalFormRequest(matrix=matrix)

    with pytest.raises(OperationDomainValidationError, match="32"):
        reduce_lattice_basis(LatticeReductionRequest.model_construct(basis=matrix))
    with pytest.raises(OperationDomainValidationError, match="32"):
        compute_hermite_normal_form(
            HermiteNormalFormRequest.model_construct(matrix=matrix)
        )

    inverse_schema = _operation(
        "matrix.inverse.compute"
    ).request_type.model_json_schema()
    square_schema = SquareIntegerMatrixRequest.model_json_schema()
    integer_schema = IntegerMatrixRequest.model_json_schema()
    lattice_schema = LatticeReductionRequest.model_json_schema()
    assert _entry_axis_limit(inverse_schema, "matrix") == 128
    assert _entry_axis_limit(square_schema, "matrix") == 32
    assert _entry_axis_limit(integer_schema, "matrix") == MAX_EXACT_LINEAR_MATRIX_AXIS
    assert _entry_axis_limit(lattice_schema, "basis") == 32
    assert IntegerMatrix.model_json_schema()["properties"]["entries"]["maxItems"] == 128
