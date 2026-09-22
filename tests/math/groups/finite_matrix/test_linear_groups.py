"""Exact contract tests for bounded prime-field GL and SL values."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.finite_matrix._models import (
    GeneralLinearNaturalActionRequest,
    PrimeFieldGeneralLinearGroup,
    PrimeFieldGeneralLinearNaturalAction,
    PrimeFieldLinearGroupRequest,
    PrimeFieldSpecialLinearNaturalAction,
    SpecialLinearNaturalActionRequest,
)
from jacobian.math.groups.finite_matrix.operations import (
    construct_general_linear_group,
    construct_special_linear_group,
    general_linear_nonzero_vector_action,
    special_linear_nonzero_vector_action,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

Matrix = tuple[tuple[int, ...], ...]


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


def test_gl_one_is_the_prime_field_unit_group() -> None:
    group = construct_general_linear_group(5, 1)

    assert group.order == 4
    assert tuple(generator.entries for generator in group.generators) == (((2,),),)
    assert _generated_matrices(group.generators, 5, 1) == {
        ((1,),),
        ((2,),),
        ((3,),),
        ((4,),),
    }


def test_gl_two_over_f2_generators_are_complete() -> None:
    group = construct_general_linear_group(2, 2)
    generated = _generated_matrices(group.generators, 2, 2)
    independently_invertible = {
        matrix for matrix in _all_two_by_two(2) if _determinant_2(matrix, 2) != 0
    }

    assert group.order == 6
    assert generated == independently_invertible


def test_gl_two_over_f3_order_and_generators_match_all_invertible_matrices() -> None:
    group = construct_general_linear_group(3, 2)
    generated = _generated_matrices(group.generators, 3, 2)
    independently_invertible = {
        matrix for matrix in _all_two_by_two(3) if _determinant_2(matrix, 3) != 0
    }

    assert group.order == 48
    assert generated == independently_invertible


def test_sl_two_over_f3_is_exactly_the_determinant_kernel() -> None:
    group = construct_special_linear_group(3, 2)
    generated = _generated_matrices(group.generators, 3, 2)
    determinant_kernel = {
        matrix for matrix in _all_two_by_two(3) if _determinant_2(matrix, 3) == 1
    }

    assert group.order == 24
    assert group.ambient_general_linear_order == 48
    assert group.determinant_index == 2
    assert generated == determinant_kernel


def test_sl_one_is_trivial_but_retains_an_action_generator() -> None:
    group = construct_special_linear_group(7, 1)

    assert group.order == 1
    assert tuple(generator.entries for generator in group.generators) == (((1,),),)


def test_composite_characteristic_and_native_scalar_types_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError) as composite:
        construct_general_linear_group(9, 2)
    assert composite.value.errors()[0]["type"] == (
        "finite_matrix_group.characteristic_not_prime"
    )
    with pytest.raises(OperationDomainValidationError):
        construct_special_linear_group(True, 2)
    with pytest.raises(OperationDomainValidationError):
        construct_special_linear_group(3, False)


def test_constructor_boundaries_are_explicit() -> None:
    boundary = construct_general_linear_group(2, 12)
    assert boundary.dimension == 12
    assert len(boundary.generators) == 22

    with pytest.raises(OperationResourceAdmissionError) as error:
        construct_general_linear_group(2, 13)
    assert error.value.errors()[0]["type"] == "finite_matrix_group.dimension_bound"


def test_gl_natural_action_is_faithful_on_nonzero_vectors() -> None:
    group = construct_general_linear_group(2, 2)
    result = general_linear_nonzero_vector_action(group)

    assert result.vectors == ((0, 1), (1, 0), (1, 1))
    generated = _generated_permutations(
        result.action.generators, len(result.action.domain)
    )
    assert len(generated) == group.order == 6
    positions = {vector: index for index, vector in enumerate(result.vectors)}
    for matrix, permutation in zip(
        group.generators, result.action.generators, strict=True
    ):
        for index, vector in enumerate(result.vectors):
            image = (
                sum(matrix.entries[0][column] * vector[column] for column in range(2))
                % 2,
                sum(matrix.entries[1][column] * vector[column] for column in range(2))
                % 2,
            )
            assert permutation[index] == positions[image]


def test_sl_natural_action_retains_field_axis_and_group_order() -> None:
    group = construct_special_linear_group(3, 2)
    result = special_linear_nonzero_vector_action(group)

    assert len(result.vectors) == 8
    assert len(_generated_permutations(result.action.generators, 8)) == group.order
    restored = PrimeFieldSpecialLinearNaturalAction.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_natural_action_rejects_forged_named_group() -> None:
    group = construct_general_linear_group(2, 2)
    forged = group.model_copy(update={"order": 7})

    with pytest.raises(OperationDomainValidationError) as error:
        general_linear_nonzero_vector_action(forged)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.authored_group_mismatch"
    )


def test_natural_action_size_is_preflighted_before_vector_enumeration() -> None:
    group = construct_general_linear_group(11, 2)

    with pytest.raises(OperationResourceAdmissionError) as error:
        general_linear_nonzero_vector_action(group)
    assert error.value.errors()[0]["type"] == (
        "finite_matrix_group.natural_action_size_bound"
    )


def test_serialized_gl_group_composes_into_natural_action() -> None:
    group = construct_general_linear_group(3, 2)
    restored = PrimeFieldGeneralLinearGroup.model_validate_json(group.model_dump_json())
    result = general_linear_nonzero_vector_action(restored)
    round_trip = PrimeFieldGeneralLinearNaturalAction.model_validate_json(
        result.model_dump_json()
    )

    assert round_trip.group == restored
    assert len(round_trip.vectors) == 8


def test_result_models_reject_mismatched_generator_parents() -> None:
    group = construct_general_linear_group(2, 2)
    payload = json.loads(group.model_dump_json())
    payload["generators"][0]["prime"] = 3

    with pytest.raises(ValidationError):
        PrimeFieldGeneralLinearGroup.model_validate_json(json.dumps(payload))


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
