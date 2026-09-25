"""Exact relative traces with independent finite-field conjugate checks."""

from __future__ import annotations

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.function_fields import (
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_element_trace,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldTraceRequest,
    FunctionFieldTraceResult,
)
from jacobian.math.function_fields._tools import TOOLS

OPERATION_ID = "function_field.element.trace.compute"


def _rational(
    prime: int, numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=prime, coefficients=numerator),
        denominator=PrimeFieldPolynomial(
            characteristic=prime, coefficients=denominator
        ),
    )


def _quadratic_y_squared_x() -> FiniteFunctionField:
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


def _evaluate_polynomial(coefficients: tuple[int, ...], point: int, prime: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * point + coefficient) % prime
    return value


def _evaluate(value: PrimeFieldRationalFunction, point: int) -> int:
    prime = value.characteristic
    numerator = _evaluate_polynomial(value.numerator.coefficients, point, prime)
    denominator = _evaluate_polynomial(value.denominator.coefficients, point, prime)
    assert denominator
    return numerator * pow(denominator, prime - 2, prime) % prime


def test_quadratic_trace_matches_sum_over_specialized_conjugates() -> None:
    field = _quadratic_y_squared_x()
    # In GF(5)(x)[y]/(y^2-x), a has two conjugates a0 +/- a1*sqrt(x).
    # Their sum is 2*a0, independently checked at each nonsingular split point.
    a0 = _rational(5, (1, 1), (1, 2))
    a1 = _rational(5, (2, 1), (2, 1))
    element = _element(field, (a0, a1))

    result = function_field_element_trace(element)

    assert result.field == field
    assert result.element.coordinates == (
        _rational(5, (3, 3), (3, 1)),
        _rational(5, (1,)),
    )
    for point in (1, 4):
        roots = tuple(root for root in range(5) if root * root % 5 == point)
        assert len(roots) == 2
        expected = (
            sum(
                (_evaluate(a0, point) + _evaluate(a1, point) * root) % 5
                for root in roots
            )
            % 5
        )
        assert _evaluate(result.trace, point) == expected


def test_trace_of_one_respects_characteristic_dividing_extension_degree() -> None:
    # y^3+y+x is separable and irreducible over GF(2)(x), with extension degree
    # 3. The trace of one is 3*1 = 1 in characteristic two.
    field = FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rational(2, (0, 1)),
            _rational(2, (1,)),
            _rational(2, (0,)),
            _rational(2, (1,)),
        ),
    )
    one = _rational(2, (1,))
    zero = _rational(2, (0,))

    result = function_field_element_trace(_element(field, (one, zero, zero)))

    assert result.trace == one


def test_rational_function_field_trace_is_identity() -> None:
    field = FiniteFunctionField(
        characteristic=7,
        variable="x",
        generator="y",
        defining_polynomial=(_rational(7, (1,)),),
    )
    value = _rational(7, (1, 1), (1, 0, 1))

    result = function_field_element_trace(_element(field, (value,)))

    assert result.trace == _rational(7, (1, 1), (1, 0, 1))


def test_trace_growth_admission_precedes_rational_function_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.function_fields.operations as operations

    field = _quadratic_y_squared_x()
    element = _element(field, (_rational(5, (1,)), _rational(5, (1,))))
    monkeypatch.setattr(
        operations, "_preflight_inverse_operand", lambda _element: (field, element)
    )
    monkeypatch.setattr(operations, "_admit_field", lambda _field: None)
    monkeypatch.setattr(operations, "MAX_TRACE_WORK", 0)

    def unexpected_expansion(*_args, **_kwargs):
        raise AssertionError("trace admission must precede rational-function expansion")

    monkeypatch.setattr(operations, "rf_mul", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_trace(element)
    assert (
        error.value.errors()[0]["type"] == "function_field.trace_work_exceeds_envelope"
    )


def test_trace_tool_schema_dispatch_and_wire_roundtrip() -> None:
    field = _quadratic_y_squared_x()
    element = _element(field, (_rational(5, (1,)), _rational(5, (1,))))
    direct = function_field_element_trace(element)
    restored = FunctionFieldTraceResult.model_validate_json(direct.model_dump_json())
    assert restored == direct

    tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    request = FunctionFieldTraceRequest(element=element)
    assert tool.run(request) == direct
    example = tool.request_type.model_validate_json(
        encode_strict_json(next(iter(tool.examples)).input), strict=True
    )
    assert tool.run(example).trace == _rational(2, (1,))
    assert any(candidate.operation_id == OPERATION_ID for candidate in TOOLS)
