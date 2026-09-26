"""Exact evidence for integral unimodular quadratic-form coordinate changes."""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.unimodular._models import (
    UnimodularChangeRequest,
)
from jacobian.math.number_theory.quadratic_forms.integral.unimodular.operations import (
    unimodular_change,
)


def _polynomial_value(form: IntegralQuadraticForm, coordinates: tuple[int, ...]) -> int:
    result = sum(
        coefficient * coordinate**2
        for coefficient, coordinate in zip(
            form.diagonal_coefficients, coordinates, strict=True
        )
    )
    return result + sum(
        term.coefficient * coordinates[term.left] * coordinates[term.right]
        for term in form.cross_terms
    )


def _matrix_vector(matrix: IntegerMatrix, vector: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(
            int(entry) * coordinate
            for entry, coordinate in zip(row, vector, strict=True)
        )
        for row in matrix.entries
    )


def _request(
    form: IntegralQuadraticForm,
    matrix: tuple[tuple[int, ...], ...],
    target_axis: tuple[str, ...],
) -> UnimodularChangeRequest:
    return UnimodularChangeRequest(
        form=form,
        matrix=IntegerMatrix(
            row_count=len(matrix),
            column_count=len(matrix),
            entries=matrix,
        ),
        target_axis=target_axis,
    )


def test_unimodular_shear_preserves_form_by_exact_coordinate_transport() -> None:
    source = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(1, 2),
        cross_terms=({"left": 0, "right": 1, "coefficient": 3},),
    )
    result = unimodular_change(_request(source, ((1, 1), (0, 1)), ("u", "v")))

    assert result.target.diagonal_coefficients == (1, 6)
    assert [
        (term.left, term.right, term.coefficient) for term in result.target.cross_terms
    ] == [(0, 1, 5)]
    for vector in itertools.product(range(-3, 4), repeat=2):
        old_coordinates = _matrix_vector(result.matrix, vector)
        assert _polynomial_value(result.target, vector) == _polynomial_value(
            source, old_coordinates
        )

    identity = tuple(
        tuple(
            sum(
                int(result.matrix.entries[row][k])
                * int(result.inverse.entries[k][column])
                for k in range(2)
            )
            for column in range(2)
        )
        for row in range(2)
    )
    assert identity == ((1, 0), (0, 1))


def test_unimodular_reflection_and_zero_dimensional_change() -> None:
    source = IntegralQuadraticForm(axis=("x",), diagonal_coefficients=(7,))
    reflection = unimodular_change(_request(source, ((-1,),), ("u",)))
    assert reflection.target.diagonal_coefficients == (7,)
    assert reflection.inverse.entries == ((-1,),)

    empty = IntegralQuadraticForm(axis=(), diagonal_coefficients=())
    identity = unimodular_change(_request(empty, (), ()))
    assert identity.target == empty
    assert identity.matrix.row_count == identity.inverse.row_count == 0


def test_three_dimensional_row_swap_and_shear() -> None:
    source = IntegralQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(2, 3, 5),
        cross_terms=(
            {"left": 0, "right": 1, "coefficient": 1},
            {"left": 1, "right": 2, "coefficient": -2},
        ),
    )
    matrix = ((0, 1, 0), (1, 0, 1), (0, 1, 1))
    result = unimodular_change(_request(source, matrix, ("a", "b", "c")))
    assert result.inverse.entries == ((1, 1, -1), (1, 0, 0), (-1, 0, 1))
    for vector in itertools.product(range(-2, 3), repeat=3):
        assert _polynomial_value(result.target, vector) == _polynomial_value(
            source, _matrix_vector(result.matrix, vector)
        )

    # The leading principal pivot at column 1 vanishes, so Bareiss must swap
    # rows a second time after its first elimination step.
    second_pivot_swap = unimodular_change(
        _request(source, ((1, 1, 0), (1, 1, 1), (0, 1, 2)), ("p", "q", "r"))
    )
    for vector in itertools.product(range(-2, 3), repeat=3):
        assert _polynomial_value(second_pivot_swap.target, vector) == _polynomial_value(
            source, _matrix_vector(second_pivot_swap.matrix, vector)
        )


def test_nonunimodular_matrix_is_rejected_before_transport() -> None:
    source = IntegralQuadraticForm(axis=("x",), diagonal_coefficients=(1,))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        unimodular_change(_request(source, ((2,),), ("u",)))
    assert exc_info.value.errors()[0]["type"] == (
        "quadratic_form.unimodular.determinant_not_unit"
    )


def test_all_small_two_by_two_determinants_match_independent_formula() -> None:
    source = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(1, 3),
        cross_terms=({"left": 0, "right": 1, "coefficient": 1},),
    )
    for entries in itertools.product(range(-1, 2), repeat=4):
        a, b, c, d = entries
        matrix = ((a, b), (c, d))
        determinant = a * d - b * c
        request = _request(source, matrix, ("u", "v"))
        if abs(determinant) != 1:
            with pytest.raises(OperationDomainValidationError):
                unimodular_change(request)
            continue

        result = unimodular_change(request)
        expected_inverse = (
            (d // determinant, -b // determinant),
            (-c // determinant, a // determinant),
        )
        assert result.inverse.entries == expected_inverse
        for vector in ((0, 0), (1, -2), (-2, 1)):
            assert _polynomial_value(result.target, vector) == _polynomial_value(
                source, _matrix_vector(result.matrix, vector)
            )


def test_coefficient_growth_is_rejected_during_preflight() -> None:
    source = IntegralQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(10**255, 0),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        unimodular_change(_request(source, ((1, 9), (0, 1)), ("u", "v")))
    assert exc_info.value.errors()[0]["type"] == (
        "quadratic_form.unimodular.coefficient_growth_bound"
    )


def test_matrix_shape_and_entry_limits_precede_nested_matrix_parse() -> None:
    form = {
        "axis": ["x"],
        "diagonal_coefficients": ["1"],
    }
    with pytest.raises(ValidationError) as shape_error:
        UnimodularChangeRequest.model_validate(
            {
                "form": form,
                "matrix": {
                    "row_count": 33,
                    "column_count": 33,
                    "entries": [["0"] * 33 for _ in range(33)],
                },
                "target_axis": ["u"],
            }
        )
    assert shape_error.value.errors()[0]["type"] == (
        "quadratic_form.unimodular.matrix_shape_bound"
    )

    with pytest.raises(ValidationError) as entry_error:
        UnimodularChangeRequest.model_validate(
            {
                "form": form,
                "matrix": {
                    "row_count": 1,
                    "column_count": 1,
                    "entries": [["1" + "0" * 128]],
                },
                "target_axis": ["u"],
            }
        )
    assert entry_error.value.errors()[0]["type"] == (
        "quadratic_form.unimodular.matrix_entry_bound"
    )


def test_advertised_tool_composes_after_json_roundtrip() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "quadratic_form.unimodular_change.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    transported = result.model_validate_json(result.model_dump_json())
    assert transported.target == result.target
    assert transported.inverse == result.inverse
    assert transported.target.diagonal_coefficients == (1, 6)
    assert transported.target.cross_terms[0].coefficient == 5
