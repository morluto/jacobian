"""Exact tests for primitive circuits of integer configurations."""

from __future__ import annotations

import json
from itertools import combinations
from math import gcd

import pytest
from pydantic import ValidationError
from sympy import Matrix

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups import integer_configuration_circuits
from jacobian.math.affine_semigroups._models import (
    IntegerConfigurationCircuitsRequest,
    IntegerConfigurationCircuitsResult,
)
from jacobian.math.affine_semigroups._tools import TOOLS, _run_circuits
from jacobian.math.matrices.values import IntegerMatrix


def _matrix(rows: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(entries=tuple(tuple(row) for row in rows))


def _primitive_sign(vector: tuple[int, ...]) -> tuple[int, ...]:
    divisor = 0
    for value in vector:
        divisor = gcd(divisor, abs(value))
    result = tuple(value // divisor for value in vector)
    return (
        tuple(-value for value in result)
        if next(x for x in result if x) < 0
        else result
    )


def _sympy_circuit_oracle(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Independent small-matrix enumeration using SymPy rational rank/nullspace."""
    n = len(rows[0])
    result: list[tuple[int, ...]] = []
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            submatrix = Matrix([[row[column] for column in support] for row in rows])
            if submatrix.rank() != size - 1:
                continue
            if any(
                Matrix(
                    [
                        [row[column] for column in support if column != removed]
                        for row in rows
                    ]
                ).rank()
                != size - 1
                for removed in support
            ):
                continue
            vector = submatrix.nullspace()[0]
            den = 1
            for value in vector:
                den = den * int(value.q) // gcd(den, int(value.q))
            primitive = _primitive_sign(tuple(int(value * den) for value in vector))
            full = [0] * n
            for column, value in zip(support, primitive, strict=True):
                full[column] = value
            result.append(tuple(full))
    return tuple(sorted(result))


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [[1, 2, 3]],
            ((0, 3, -2), (2, -1, 0), (3, 0, -1)),
        ),
        (
            [[1, 0, -1], [0, 1, -1]],
            ((1, 1, 1),),
        ),
        (
            [[0, 1, 1]],
            ((0, 1, -1), (1, 0, 0)),
        ),
        ([[1, 0], [0, 1]], ()),
    ],
)
def test_circuits_have_expected_primitive_supports(
    rows: list[list[int]], expected: tuple[tuple[int, ...], ...]
) -> None:
    result = integer_configuration_circuits(_matrix(rows))
    assert result.circuits == expected
    assert result.circuits == _sympy_circuit_oracle(rows)


def test_each_result_replays_and_has_minimal_support() -> None:
    rows = [[1, 0, 1, 2], [0, 1, 1, 3]]
    result = integer_configuration_circuits(_matrix(rows))
    for vector in result.circuits:
        assert all(
            sum(row[column] * vector[column] for column in range(len(vector))) == 0
            for row in rows
        )
        assert gcd(*(abs(value) for value in vector)) == 1
        support = tuple(index for index, value in enumerate(vector) if value)
        for removed in support:
            proper = tuple(index for index in support if index != removed)
            assert Matrix(
                [[row[index] for index in proper] for row in rows]
            ).rank() == len(proper)


def test_circuits_are_covariant_under_row_unimodular_and_column_permutations() -> None:
    original = [[1, 0, -1], [0, 1, -1]]
    transformed = [[2, 1, -3], [1, 1, -2]]  # [[2,1],[1,1]] has determinant 1.
    assert integer_configuration_circuits(_matrix(original)).circuits == (
        integer_configuration_circuits(_matrix(transformed)).circuits
    )
    permuted = [[row[index] for index in (2, 0, 1)] for row in original]
    actual = integer_configuration_circuits(_matrix(permuted)).circuits
    transported = tuple(
        sorted((vector[1], vector[2], vector[0]) for vector in ((1, 1, 1),))
    )
    assert actual == transported


def test_catalog_example_and_serialized_result_replay() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "integer_configuration.circuits.compute"
    )
    request = IntegerConfigurationCircuitsRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(request)
    assert isinstance(result, IntegerConfigurationCircuitsResult)
    assert _run_circuits(request) == result
    decoded = IntegerConfigurationCircuitsResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert decoded.configuration.column_count == 3


def test_full_admitted_axis_and_input_limits_are_enforced() -> None:
    identity = _matrix(
        [[int(row == column) for column in range(12)] for row in range(12)]
    )
    assert integer_configuration_circuits(identity).circuits == ()
    too_many_columns = _matrix([[1] * 13])
    with pytest.raises(OperationResourceAdmissionError):
        integer_configuration_circuits(too_many_columns)
    with pytest.raises(OperationResourceAdmissionError):
        integer_configuration_circuits(_matrix([[10**8, 1]]))
    with pytest.raises(ValidationError):
        IntegerConfigurationCircuitsRequest.model_validate(
            {"configuration": {"entries": [[str(10**8), "1"]]}}
        )


def test_forged_native_matrix_is_bounded_before_revalidation_copy() -> None:
    forged = IntegerMatrix.model_construct(
        domain="ZZ",
        row_count=1,
        column_count=1,
        entries=((1 << 1_000_000,),),
    )
    with pytest.raises(OperationResourceAdmissionError):
        integer_configuration_circuits(forged)


def test_result_rejects_zero_misshaped_or_misbound_circuit_vectors() -> None:
    config = _matrix([[1, 1]])
    for circuits in (((0, 0),), ((1, 0),), ((1,),)):
        with pytest.raises(ValidationError):
            IntegerConfigurationCircuitsResult.model_validate(
                {"configuration": config.model_dump(mode="json"), "circuits": circuits}
            )
