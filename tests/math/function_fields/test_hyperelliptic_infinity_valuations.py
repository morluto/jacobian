import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldFiniteValuation,
    FunctionFieldPositiveInfinityValuation,
    HyperellipticInfinityPlace,
    HyperellipticInfinityPlaceValuationRequest,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_element_multiply,
    function_field_hyperelliptic_infinity_valuation,
)


def _rf(numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)):
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=denominator),
    )


def _field(branch: tuple[int, ...] = (0, 1, 0, 4)) -> FiniteFunctionField:
    # y^2 = x^3-x by default; the quintic fixture is also smooth.
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rf(tuple(-v % 5 for v in branch)), _rf((0,)), _rf((1,))),
    )


def _place(field: FiniteFunctionField | None = None) -> HyperellipticInfinityPlace:
    field = field or _field()
    return HyperellipticInfinityPlace(
        field=field,
        residue_field=FiniteFieldPresentation(
            characteristic=5, modulus_coefficients=(0, 1), generator="z"
        ),
    )


def _element(c0: tuple[int, ...], c1: tuple[int, ...] = (0,), den0=(1,), den1=(1,)):
    field = _field()
    return FiniteFunctionFieldElement(
        field=field,
        coordinates=(_rf(c0, den0), _rf(c1, den1)),
    )


@pytest.mark.parametrize(
    ("element", "expected"),
    [
        (_element((1,)), 0),
        (_element((0, 1)), -2),
        (_element((0,), (1,)), -3),
        (_element((1,), den0=(0, 1)), 2),
        (_element((0, 1), (1,)), -3),
    ],
)
def test_odd_degree_infinity_valuations(element, expected):
    result = function_field_hyperelliptic_infinity_valuation(_place(), element)
    assert isinstance(result.valuation, FunctionFieldFiniteValuation)
    assert result.valuation.value == expected
    assert result.place.field == result.element.field
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_zero_has_structural_infinite_valuation_and_degree_five_uses_its_degree():
    zero = function_field_hyperelliptic_infinity_valuation(_place(), _element((0,)))
    assert isinstance(zero.valuation, FunctionFieldPositiveInfinityValuation)
    assert zero.valuation.model_dump() == {"kind": "POSITIVE_INFINITY"}

    quintic = _field((1, 0, 1, 0, 0, 1))
    y = FiniteFunctionFieldElement(field=quintic, coordinates=(_rf((0,)), _rf((1,))))
    value = function_field_hyperelliptic_infinity_valuation(_place(quintic), y)
    assert value.valuation == FunctionFieldFiniteValuation(kind="FINITE", value=-5)


def test_valuation_is_multiplicative_and_operation_is_discoverable():
    y = _element((0,), (1,))
    square = function_field_element_multiply(y, y).product
    assert (
        function_field_hyperelliptic_infinity_valuation(
            _place(), square
        ).valuation.value
        == -6
    )
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "function_field.hyperelliptic_infinity_place.valuation.compute"
    )
    assert (
        tool.run(
            tool.request_type.model_validate({"place": _place(), "element": y})
        ).valuation.value
        == -3
    )


def test_even_degree_models_are_not_misrepresented_as_one_rational_place():
    even_degree = _field((1, 0, 1, 0, 1))
    element = FiniteFunctionFieldElement(
        field=even_degree, coordinates=(_rf((1,)), _rf((0,)))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_hyperelliptic_infinity_valuation(_place(even_degree), element)
    assert error.value.errors()[0]["type"] == (
        "function_field.odd_hyperelliptic_infinity_required"
    )


def test_native_request_allows_equivalent_unreduced_parent_spellings():
    field = _field()
    equivalent = FiniteFunctionField.model_construct(
        **{
            **field.model_dump(),
            "defining_polynomial": (
                PrimeFieldRationalFunction.model_construct(
                    numerator=PrimeFieldPolynomial.model_construct(
                        characteristic=5, coefficients=(1,)
                    ),
                    denominator=PrimeFieldPolynomial.model_construct(
                        characteristic=5, coefficients=(1, 0)
                    ),
                ),
                *field.defining_polynomial[1:],
            ),
        }
    )
    element = FiniteFunctionFieldElement(
        field=equivalent, coordinates=(_rf((1,)), _rf((0,)))
    )
    request = HyperellipticInfinityPlaceValuationRequest(
        place=_place(field), element=element
    )
    assert request.place.field != request.element.field
