"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/finite_matrix/test_extension_linear_groups.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from itertools import product

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.finite_fields.values import Axis, FiniteFieldPresentation


def _add(left: int, right: int) -> int:
    return left ^ right


def _multiply_field(left: int, right: int) -> int:
    a0, a1 = left & 1, (left >> 1) & 1
    b0, b1 = right & 1, (right >> 1) & 1
    c0 = (a0 & b0) ^ (a1 & b1)
    c1 = (a0 & b1) ^ (a1 & b0) ^ (a1 & b1)
    return c0 + 2 * c1


def _multiply_matrix(left: Matrix, right: Matrix) -> Matrix:
    dimension = len(left)
    return tuple(
        tuple(
            _add(
                _multiply_field(left[row][0], right[0][column]),
                _multiply_field(left[row][1], right[1][column]),
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _determinant_2(matrix: Matrix) -> int:
    return _add(
        _multiply_field(matrix[0][0], matrix[1][1]),
        _multiply_field(matrix[0][1], matrix[1][0]),
    )


def _generator_matrices(group: object) -> tuple[Matrix, ...]:
    return tuple(
        tuple(
            tuple(
                element.coordinates[0] + 2 * element.coordinates[1] for element in row
            )
            for row in generator.entries
        )
        for generator in group.generators  # type: ignore[attr-defined]
    )


def _generated_matrices(generators: tuple[Matrix, ...]) -> set[Matrix]:
    identity = ((1, 0), (0, 1))
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = _multiply_matrix(generator, current)
            if candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen


def _all_two_by_two() -> tuple[Matrix, ...]:
    return tuple(((a, b), (c, d)) for a, b, c, d in product(range(4), repeat=4))


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


def _coordinates(value: object) -> int:
    return value.coordinates[0] + 2 * value.coordinates[1]  # type: ignore[attr-defined]


def _projective_image(matrix: Matrix, point: tuple[int, int]) -> tuple[int, int]:
    image = (
        _add(
            _multiply_field(matrix[0][0], point[0]),
            _multiply_field(matrix[0][1], point[1]),
        ),
        _add(
            _multiply_field(matrix[1][0], point[0]),
            _multiply_field(matrix[1][1], point[1]),
        ),
    )
    pivot = next(value for value in image if value)
    inverse = {1: 1, 2: 3, 3: 2}[pivot]
    return tuple(_multiply_field(value, inverse) for value in image)  # type: ignore[return-value]


def test_catalog_examples_validate_and_execute() -> None:
    operation_ids = {
        "finite_matrix_group.extension.general_linear.construct",
        "finite_matrix_group.extension.special_linear.construct",
        "finite_matrix_group.extension.general_linear.projective_action.compute",
        "finite_matrix_group.extension.special_linear.projective_action.compute",
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


GF4 = FiniteFieldPresentation(
    characteristic=2, modulus_coefficients=(1, 1, 1), generator="a"
)
AXIS = Axis(name="V", labels=("x", "y"))
Matrix = tuple[tuple[int, ...], ...]
