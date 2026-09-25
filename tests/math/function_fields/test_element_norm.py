"""Exact function-field norms with independent determinant checks."""

from __future__ import annotations

from jacobian.canonical import encode_strict_json
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
