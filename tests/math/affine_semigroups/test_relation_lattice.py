"""Exact contract tests for integer configuration relation lattices."""

from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups._models import (
    MAX_RELATION_LATTICE_DIMENSION,
    MAX_RELATION_LATTICE_INPUT_DIGITS,
    RelationLatticeRequest,
    RelationLatticeResult,
)
from jacobian.math.affine_semigroups._tools import TOOLS, _run_relation_lattice
from jacobian.math.affine_semigroups.operations import relation_lattice
from jacobian.math.matrices.values import IntegerMatrix


def _matrix(entries: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(entries=tuple(tuple(row) for row in entries))


def _product(configuration: IntegerMatrix, basis: IntegerMatrix) -> list[list[int]]:
    rows = configuration.entries
    return [
        [
            sum(int(a) * int(b) for a, b in zip(row, generator, strict=True))
            for generator in basis.entries
        ]
        for row in rows
    ]


def test_one_row_configuration_has_rank_two_kernel() -> None:
    result = relation_lattice(_matrix([[1, 2, 3]]))

    assert result.rank == 1
    assert result.nullity == 2
    assert result.smith_rank == 1
    assert tuple(int(value) for value in result.smith_invariant_factors) == (1,)
    assert result.is_saturated
    assert result.saturation_index == 1
    assert result.relation_basis.row_count == 2
    assert result.relation_basis.column_count == 3
    assert _product(result.configuration, result.relation_basis) == [[0, 0]]
    assert result.relation_lattice.ambient_dimension == 3


def test_full_rank_configuration_has_trivial_kernel() -> None:
    result = relation_lattice(_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    assert result.rank == 3
    assert result.nullity == 0
    assert result.relation_basis.row_count == 0
    assert result.relation_lattice.basis.row_count == 0


def test_known_kernel_generator_is_spanned() -> None:
    configuration = _matrix([[1, 1, 2], [0, 1, 1]])

    result = relation_lattice(configuration)

    assert result.rank == 2
    assert result.nullity == 1
    assert _product(result.configuration, result.relation_basis) == [[0], [0]]
    row = tuple(int(value) for value in result.relation_basis.entries[0])
    assert row in {(1, 1, -1), (-1, -1, 1)}


def test_zero_configuration_kernel_is_the_full_lattice() -> None:
    result = relation_lattice(_matrix([[0, 0], [0, 0]]))

    assert result.rank == 0
    assert result.nullity == 2
    assert _product(result.configuration, result.relation_basis) == [[0, 0], [0, 0]]


def test_smith_invariant_factors_record_elementary_divisors() -> None:
    result = relation_lattice(_matrix([[2, 0], [0, 4]]))

    assert tuple(int(value) for value in result.smith_invariant_factors) == (2, 4)
    assert result.rank == 2
    assert result.nullity == 0


def test_result_round_trips_through_serialization() -> None:
    result = relation_lattice(_matrix([[1, 1, 2], [0, 1, 1]]))
    decoded = RelationLatticeResult.model_validate_json(result.model_dump_json())

    assert decoded == result


def test_oversized_scalars_are_rejected_at_native_admission() -> None:
    oversized = 10**MAX_RELATION_LATTICE_INPUT_DIGITS

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        relation_lattice(_matrix([[oversized, 1], [1, 0]]))

    assert "digits" in excinfo.value.errors()[0]["msg"]


def test_oversized_axes_are_rejected_at_native_admission() -> None:
    order = MAX_RELATION_LATTICE_DIMENSION + 1
    entries = [
        [1 if column == row else 0 for column in range(order)] for row in range(order)
    ]

    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        relation_lattice(_matrix(entries))

    assert "rows and columns" in excinfo.value.errors()[0]["msg"]


def test_native_and_catalog_paths_agree() -> None:
    configuration = _matrix([[1, 2, 3]])
    request = RelationLatticeRequest(configuration=configuration)

    assert _run_relation_lattice(request) == relation_lattice(configuration)


def test_declaration_is_published_with_one_executable_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "integer_configuration.relation_lattice.compute"
    )

    assert len(tool.examples) >= 1
    example_request = RelationLatticeRequest.model_validate_json(
        json.dumps(dict(tool.examples[0].input))
    )
    result = tool.run(example_request)
    assert isinstance(result, RelationLatticeResult)
    assert result.nullity == 2
