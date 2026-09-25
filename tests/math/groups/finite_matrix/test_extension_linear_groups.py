"""Exact composition tests for bounded extension-field GL and SL values."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

import jacobian.math.groups.finite_matrix.extension_operations as extension_operations
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import Axis, FiniteFieldPresentation
from jacobian.math.groups.actions.operations import cycle_index
from jacobian.math.groups.finite_matrix._models import (
    ExtensionFieldGeneralLinearGroup,
    ExtensionFieldGeneralLinearProjectiveAction,
    ExtensionFieldLinearGroupRequest,
    ExtensionFieldSpecialLinearProjectiveAction,
)
from jacobian.math.groups.finite_matrix.extension_operations import (
    construct_extension_general_linear_group,
    construct_extension_special_linear_group,
    extension_general_linear_projective_action,
    extension_special_linear_projective_action,
)

GF4 = FiniteFieldPresentation(
    characteristic=2, modulus_coefficients=(1, 1, 1), generator="a"
)
AXIS = Axis(name="V", labels=("x", "y"))
Matrix = tuple[tuple[int, ...], ...]


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


def test_extension_gl_and_sl_generators_match_independent_gf4_enumeration() -> None:
    gl = construct_extension_general_linear_group(GF4, AXIS)
    sl = construct_extension_special_linear_group(GF4, AXIS)
    all_matrices = _all_two_by_two()
    invertible = {matrix for matrix in all_matrices if _determinant_2(matrix) != 0}
    determinant_one = {matrix for matrix in all_matrices if _determinant_2(matrix) == 1}

    assert gl.order == 180
    assert sl.order == 60
    assert sl.ambient_general_linear_order == gl.order
    assert sl.determinant_index == 3
    assert _generated_matrices(_generator_matrices(gl)) == invertible
    assert _generated_matrices(_generator_matrices(sl)) == determinant_one


def test_projective_actions_match_direct_coordinates_and_compose_after_json() -> None:
    gl = construct_extension_general_linear_group(GF4, AXIS)
    sl = construct_extension_special_linear_group(GF4, AXIS)
    gl_action = extension_general_linear_projective_action(gl)
    sl_action = extension_special_linear_projective_action(sl)
    points = tuple(
        tuple(_coordinates(value) for value in point.coordinates)
        for point in gl_action.points
    )
    positions = {point: index for index, point in enumerate(points)}

    assert points == ((1, 0), (1, 1), (1, 2), (1, 3), (0, 1))
    for matrix, permutation in zip(
        _generator_matrices(gl), gl_action.action.generators, strict=True
    ):
        assert permutation == tuple(
            positions[_projective_image(matrix, point)] for point in points
        )
    assert len(_generated_permutations(gl_action.action.generators, 5)) == 60
    assert len(_generated_permutations(sl_action.action.generators, 5)) == 60

    restored_gl = ExtensionFieldGeneralLinearGroup.model_validate_json(
        gl.model_dump_json()
    )
    restored_action = ExtensionFieldGeneralLinearProjectiveAction.model_validate_json(
        extension_general_linear_projective_action(restored_gl).model_dump_json()
    )
    assert cycle_index(restored_action.action).group_order == 60
    assert restored_action.group == restored_gl

    restored_sl = ExtensionFieldSpecialLinearProjectiveAction.model_validate_json(
        sl_action.model_dump_json()
    )
    assert cycle_index(restored_sl.action).group_order == 60


def test_extension_group_consumers_reject_forged_order_and_generators() -> None:
    group = construct_extension_general_linear_group(GF4, AXIS)
    with pytest.raises(OperationDomainValidationError) as wrong_order:
        extension_general_linear_projective_action(
            group.model_copy(update={"order": 181})
        )
    assert wrong_order.value.errors()[0]["type"] == (
        "finite_matrix_group.authored_group_mismatch"
    )

    altered_generators = (group.generators[1], *group.generators[1:])
    with pytest.raises(OperationDomainValidationError) as wrong_generators:
        extension_general_linear_projective_action(
            group.model_copy(update={"generators": altered_generators})
        )
    assert wrong_generators.value.errors()[0]["type"] == (
        "finite_matrix_group.authored_group_mismatch"
    )

    payload = json.loads(group.model_dump_json())
    payload["generators"][0]["entries"] = [[], [], []]
    with pytest.raises(ValidationError) as oversized_matrix:
        ExtensionFieldGeneralLinearGroup.model_validate_json(json.dumps(payload))
    assert oversized_matrix.value.errors()[0]["type"] == (
        "finite_matrix_group.generator_shape_mismatch"
    )


def test_extension_group_native_constructor_checks_input_types() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        construct_extension_general_linear_group(None, AXIS)  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "finite_matrix_group.presentation_type"


def test_extension_projective_action_bound_is_checked_before_enumeration() -> None:
    axis = Axis(name="W", labels=("a", "b", "c", "d"))
    group = construct_extension_general_linear_group(GF4, axis)

    with pytest.raises(OperationResourceAdmissionError) as error:
        extension_general_linear_projective_action(group)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.projective_action_size_bound"
    )

    too_many_generators = Axis(
        name="U", labels=tuple(f"u{index}" for index in range(12))
    )
    with pytest.raises(OperationResourceAdmissionError) as generator_error:
        construct_extension_general_linear_group(GF4, too_many_generators)
    assert generator_error.value.errors()[0]["type"] == (
        "finite_matrix_group.generator_output_bound"
    )


def test_long_axis_labels_bound_serialized_group_before_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = Axis(
        name="V",
        labels=tuple(f"{index}-" + "x" * 15_000 for index in range(10)),
    )

    def generators_must_not_be_built(*_args: object) -> object:
        raise AssertionError("the generator family was materialized before admission")

    monkeypatch.setattr(
        extension_operations, "_general_generators", generators_must_not_be_built
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        construct_extension_general_linear_group(GF4, axis)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.serialized_group_output_bound"
    )


def test_long_axis_labels_bound_projective_result_before_point_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "x" * 400_000
    axis = Axis(name="V", labels=(prefix + "a", prefix + "b"))
    group = construct_extension_general_linear_group(GF4, axis)

    def points_must_not_be_built(*_args: object) -> object:
        raise AssertionError("projective points were expanded before admission")

    monkeypatch.setattr(
        extension_operations, "_projective_points", points_must_not_be_built
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        extension_general_linear_projective_action(group)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.serialized_projective_action_output_bound"
    )


def test_extension_group_request_rejects_prime_field_duplicate_carrier() -> None:
    prime_field = FiniteFieldPresentation(
        characteristic=5, modulus_coefficients=(0, 1), generator="a"
    )
    with pytest.raises(ValidationError) as error:
        ExtensionFieldLinearGroupRequest(presentation=prime_field, vector_axis=AXIS)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.extension_field_required"
    )


def test_extension_gl_uses_primitive_element_search_and_handles_projective_point() -> (
    None
):
    gf9 = FiniteFieldPresentation(
        characteristic=3, modulus_coefficients=(1, 0, 1), generator="a"
    )
    one_axis = Axis(name="L", labels=("v",))
    group = construct_extension_general_linear_group(gf9, one_axis)
    action = extension_general_linear_projective_action(group)
    generator = group.generators[0].entries[0][0]
    encoded = generator.coordinates[0] + 3 * generator.coordinates[1]

    def multiply_gf9(left: int, right: int) -> int:
        a0, a1 = left % 3, left // 3
        b0, b1 = right % 3, right // 3
        return ((a0 * b0 - a1 * b1) % 3) + 3 * ((a0 * b1 + a1 * b0) % 3)

    power = 1
    generator_order = 0
    for exponent in range(1, 9):
        power = multiply_gf9(power, encoded)
        if power == 1:
            generator_order = exponent
            break

    assert generator_order == group.order == 8
    assert group.generators[0].entries[0][0].coordinates != (0, 1)
    assert len(action.points) == 1
    assert action.action.generators == ((0,),)


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
