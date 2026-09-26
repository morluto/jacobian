"""Exact function-field norms with independent determinant checks."""

from __future__ import annotations

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.function_fields import (
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_element_norm,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldNormRequest,
    FunctionFieldNormResult,
)
from jacobian.math.function_fields._tools import TOOLS

OPERATION_ID = "function_field.element.norm.compute"


def _rational(
    prime: int, numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=prime, coefficients=numerator),
        denominator=PrimeFieldPolynomial(
            characteristic=prime, coefficients=denominator
        ),
    )


def _field() -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rational(5, (0, 4)),
            _rational(5, (0,)),
            _rational(5, (1,)),
        ),
    )


def _element(
    field: FiniteFunctionField,
    coordinates: tuple[PrimeFieldRationalFunction, ...],
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=coordinates)


def test_quadratic_norm_matches_independent_multiplication_matrix() -> None:
    field = _field()
    one = _rational(5, (1,))

    # For y^2=x, multiplication by a+b*y has matrix [[a,b*x],[b,a]],
    # hence determinant a^2-b^2*x. This formula is independent of the kernel.
    result = function_field_element_norm(_element(field, (one, one)))

    assert result.norm == _rational(5, (1, 4))
    assert result.field == field
    assert result.element.coordinates == (one, one)


def test_norm_skips_unused_final_column_growth_and_canonicalizes_field() -> None:
    x12 = _rational(5, (0,) * 12 + (1,))
    field = FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rational(5, (0, 1)), x12, _rational(5, (1,))),
    )
    y = _element(field, (_rational(5, (0,)), _rational(5, (1,))))
    assert function_field_element_norm(y).norm == _rational(5, (0, 1))

    noncanonical_one = _rational(5, (1,) + (0,) * 10 + (1,), (1,) + (0,) * 10 + (1,))
    noncanonical_x = _rational(5, (0, 1) + (0,) * 10 + (1,), (1,) + (0,) * 10 + (1,))
    canonicalized_field = FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(noncanonical_x, noncanonical_one, _rational(5, (1,))),
    )
    element = _element(
        canonicalized_field,
        (_rational(5, (1,)), _rational(5, (1,))),
    )
    assert function_field_element_norm(element).norm == _rational(5, (0, 1))


def test_norm_allows_large_irrelevant_required_matrix_entry() -> None:
    # Eisenstein cubic: the third column is required to form multiplication by
    # y^2, but its degree-13 entries cancel from the determinant (N(y^2)=x^2).
    x12 = _rational(5, (0,) * 12 + (1,))
    field = FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rational(5, (0, 1)),
            _rational(5, (0,)),
            x12,
            _rational(5, (1,)),
        ),
    )
    y_squared = _element(
        field, (_rational(5, (0,)), _rational(5, (0,)), _rational(5, (1,)))
    )
    assert function_field_element_norm(y_squared).norm == _rational(5, (0, 0, 1))


def test_over_degree_norm_is_rejected_before_public_carrier_construction() -> None:
    field = _field()
    x7 = _rational(5, (0,) * 7 + (1,))
    element = _element(field, (x7, _rational(5, (0,))))

    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_norm(element)
    assert (
        error.value.errors()[0]["type"]
        == "function_field.norm_coefficient_growth_exceeds_envelope"
    )


def test_generator_norm_and_rational_field_identity() -> None:
    field = _field()
    zero = _rational(5, (0,))
    one = _rational(5, (1,))
    assert function_field_element_norm(_element(field, (zero, one))).norm == _rational(
        5, (0, 4)
    )

    rational = FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(one,),
    )
    value = _rational(5, (1, 1), (1, 0, 1))
    assert function_field_element_norm(_element(rational, (value,))).norm == value


def test_norm_tool_schema_dispatch_and_wire_roundtrip() -> None:
    field = _field()
    element = _element(field, (_rational(5, (1,)), _rational(5, (1,))))
    direct = function_field_element_norm(element)
    assert (
        FunctionFieldNormResult.model_validate_json(direct.model_dump_json()) == direct
    )
    tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    request = FunctionFieldNormRequest(element=element)
    assert tool.run(request) == direct
    example = tool.request_type.model_validate_json(
        encode_strict_json(next(iter(tool.examples)).input), strict=True
    )
    assert tool.run(example).norm == _rational(5, (1, 4))
