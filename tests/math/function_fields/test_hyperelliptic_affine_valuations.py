import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields.values import FiniteFieldPresentation
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldFiniteValuation,
    FunctionFieldPlace,
    FunctionFieldPlaceValuationRequest,
    FunctionFieldPositiveInfinityValuation,
    HyperellipticAffinePlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields._tools import TOOLS
from jacobian.math.function_fields.operations import (
    function_field_element_multiply,
    function_field_hyperelliptic_affine_valuation,
)


def _rf(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=coefficients),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
    )


def _rational(
    numerator: tuple[int, ...], denominator: tuple[int, ...]
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=denominator),
    )


def _field() -> FiniteFunctionField:
    # The defining polynomial is y^2 - (x^3-x).
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0, 1, 0, 4)), _rf((0,)), _rf((1,))),
    )


def _place(x: int, y: int) -> HyperellipticAffinePlace:
    field = _field()
    return HyperellipticAffinePlace(
        field=field,
        x=x,
        y=y,
        local_parameter="y" if y == 0 else "x_minus_x0",
        residue_field=FiniteFieldPresentation(
            characteristic=5, modulus_coefficients=(0, 1), generator="a"
        ),
    )


def _element(c0: tuple[int, ...], c1: tuple[int, ...] = (0,)) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=_field(), coordinates=(_rf(c0), _rf(c1)))


@pytest.mark.parametrize(
    ("point", "coordinates", "expected"),
    [
        ((0, 0), ((1,), (0,)), 0),  # Nonzero constants have finite order zero.
        ((0, 0), ((0,), (1,)), 1),  # y is a uniformizer at a simple branch point.
        ((0, 0), ((0, 1), (0,)), 2),  # x has order two there.
        ((2, 1), ((3, 1), (0,)), 1),  # x-2 at an unramified rational point.
        ((2, 1), ((4,), (1,)), 1),  # y-1 has a simple zero at this point.
    ],
)
def test_affine_valuation_at_branch_and_unramified_points(point, coordinates, expected):
    result = function_field_hyperelliptic_affine_valuation(
        _place(*point), _element(*coordinates)
    )
    assert isinstance(result.valuation, FunctionFieldFiniteValuation)
    assert result.valuation.kind == "FINITE"
    assert result.valuation.value == expected
    assert result.place.field == _field()
    assert result.place.residue_field.order == 5
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_affine_valuation_canonicalizes_equivalent_shared_parent_presentations():
    canonical = _field()
    scaled_field = canonical.model_copy(
        update={
            "defining_polynomial": (
                _rational((0, 2, 0, 3), (2,)),
                _rf((0,)),
                _rf((1,)),
            )
        }
    )
    place = _place(0, 0).model_copy(update={"field": scaled_field})
    element = _element((0,), (1,)).model_copy(update={"field": scaled_field})

    result = function_field_hyperelliptic_affine_valuation(place, element)

    assert result.place.field == canonical
    assert result.element.field == canonical
    assert result.valuation.value == 1


def test_affine_valuation_is_additive_on_products_and_handles_poles():
    place = _place(0, 0)
    y = _element((0,), (1,))
    x = _element((0, 1))
    product = function_field_element_multiply(y, y).product
    product_order = function_field_hyperelliptic_affine_valuation(
        place, product
    ).valuation.value
    assert product_order == 2
    assert product_order == 2 * function_field_hyperelliptic_affine_valuation(
        place, y
    ).valuation.value
    assert (
        function_field_hyperelliptic_affine_valuation(place, x).valuation.value
        == 2
    )
    inverse_x = FiniteFunctionFieldElement(
        field=_field(), coordinates=(_rational((1,), (0, 1)), _rf((0,)))
    )
    assert function_field_hyperelliptic_affine_valuation(
        place, inverse_x
    ).valuation.value == -2


def test_zero_element_returns_structural_positive_infinity_without_null():
    result = function_field_hyperelliptic_affine_valuation(
        _place(0, 0), _element((0,), (0,))
    )

    assert isinstance(result.valuation, FunctionFieldPositiveInfinityValuation)
    assert result.valuation.kind == "POSITIVE_INFINITY"
    assert result.valuation.model_dump() == {"kind": "POSITIVE_INFINITY"}
    assert '"valuation":{"kind":"POSITIVE_INFINITY"}' in result.model_dump_json()
    assert "null" not in result.model_dump_json()


def test_unramified_local_series_finds_higher_order_cancellation():
    # At (0,1) on y^2=1+x^3, f'(0)=0 while the point is still unramified
    # for x. The relation 2(y-1)+(y-1)^2=x^3 gives v(y-1)=3.
    field = FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((4, 0, 0, 4)), _rf((0,)), _rf((1,))),
    )
    place = HyperellipticAffinePlace(
        field=field,
        x=0,
        y=1,
        local_parameter="x_minus_x0",
        residue_field=FiniteFieldPresentation(
            characteristic=5, modulus_coefficients=(0, 1), generator="a"
        ),
    )
    y_minus_one = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(4,)),
                denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
            ),
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
            ),
        ),
    )

    assert (
        function_field_hyperelliptic_affine_valuation(place, y_minus_one).valuation.value
        == 3
    )


def test_affine_place_serialization_retains_parent_uniformizer_and_residue_field():
    place = _place(0, 0)
    restored = HyperellipticAffinePlace.model_validate_json(place.model_dump_json())
    assert restored == place
    assert restored.local_parameter == "y"
    assert restored.residue_field.modulus_coefficients == (0, 1)


def test_affine_valuation_rejects_nonpoint_and_nonhyperelliptic_parents():
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_hyperelliptic_affine_valuation(_place(2, 2), _element((0, 1)))
    assert error.value.errors()[0]["type"] == "function_field.affine_point_not_on_curve"

    rational = FiniteFunctionField(
        characteristic=5,
        defining_polynomial=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
                denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
            ),
        ),
    )
    rational_place = _place(0, 0).model_copy(update={"field": rational})
    rational_element = FiniteFunctionFieldElement(field=rational, coordinates=(_rf((1,)),))
    with pytest.raises(OperationDomainValidationError) as error:
        function_field_hyperelliptic_affine_valuation(rational_place, rational_element)
    assert error.value.errors()[0]["type"] == (
        "function_field.affine_place_requires_hyperelliptic_model"
    )


def test_catalog_declares_affine_hyperelliptic_valuation():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "function_field.hyperelliptic_affine_place.valuation.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert isinstance(result.valuation, FunctionFieldFiniteValuation)
    assert result.valuation.value == 1
    assert tool in BUILTIN_TOOLS


def test_rational_place_valuation_uses_same_structural_zero_result():
    rational_field = FiniteFunctionField(
        characteristic=5, defining_polynomial=(_rf((1,)),)
    )
    place = FunctionFieldPlace(
        field=rational_field,
        kind="FINITE",
        prime_polynomial=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
        degree=1,
    )
    zero = FiniteFunctionFieldElement(field=rational_field, coordinates=(_rf((0,)),))
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "function_field.place.valuation.compute"
    )
    result = tool.run(FunctionFieldPlaceValuationRequest(place=place, element=zero))

    assert isinstance(result.valuation, FunctionFieldPositiveInfinityValuation)
    assert result.model_dump()["valuation"] == {"kind": "POSITIVE_INFINITY"}
    assert "null" not in result.model_dump_json()
