"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/finite_matrix/test_linear_groups.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from itertools import product

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups.finite_matrix._models import (
    GeneralLinearNaturalActionRequest,
    PrimeFieldLinearGroupRequest,
    SpecialLinearNaturalActionRequest,
)
from jacobian.math.groups.finite_matrix.operations import (
    construct_general_linear_group,
    construct_special_linear_group,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _multiply(left: Matrix, right: Matrix, prime: int) -> Matrix:
    dimension = len(left)
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(dimension))
            % prime
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _generated_matrices(
    generators: tuple[PrimeFieldMatrix, ...], prime: int, dimension: int
) -> set[Matrix]:
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(dimension))
        for row in range(dimension)
    )
    concrete = tuple(generator.entries for generator in generators)
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in concrete:
            candidate = _multiply(generator, current, prime)
            if candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen


def _determinant_2(matrix: Matrix, prime: int) -> int:
    return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) % prime


def _all_two_by_two(prime: int) -> tuple[Matrix, ...]:
    return tuple(((a, b), (c, d)) for a, b, c, d in product(range(prime), repeat=4))


def _generated_permutations(
    generators: tuple[tuple[int, ...], ...], degree: int
) -> set[tuple[int, ...]]:
    identity = tuple(range(degree))
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = tuple(generator[current[index]] for index in range(degree))
            if candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen


def test_catalog_examples_validate_and_execute() -> None:
    operation_ids = {
        "finite_matrix_group.general_linear.construct",
        "finite_matrix_group.special_linear.construct",
        "finite_matrix_group.general_linear.nonzero_vector_action.compute",
        "finite_matrix_group.special_linear.nonzero_vector_action.compute",
    }
    tools = {
        tool.operation_id: tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id in operation_ids
    }

    assert set(tools) == operation_ids
    for tool in tools.values():
        example = tool.examples[0]
        request = tool.request_type.model_validate_json(json.dumps(example.input))
        assert isinstance(tool.run(request), tool.result_type)

    assert PrimeFieldLinearGroupRequest(prime=2, dimension=2)
    assert GeneralLinearNaturalActionRequest(group=construct_general_linear_group(2, 2))
    assert SpecialLinearNaturalActionRequest(group=construct_special_linear_group(3, 2))


Matrix = tuple[tuple[int, ...], ...]
