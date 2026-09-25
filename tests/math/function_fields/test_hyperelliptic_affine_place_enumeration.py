from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields._tools import TOOLS
from jacobian.math.function_fields.hyperelliptic_affine_places import (
    HyperellipticAffinePlacesRequest,
    enumerate_hyperelliptic_affine_places,
)


def _rf(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=coefficients),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
    )


def _field() -> FiniteFunctionField:
    # y^2 = x^3-x over GF(5), with three branch x-coordinates.
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0, 1, 0, 4)), _rf((0,)), _rf((1,))),
    )


def test_enumeration_is_complete_canonical_and_composes_with_valuation():
    field = _field()
    result = enumerate_hyperelliptic_affine_places(field)
    coordinates = tuple((place.x, place.y) for place in result.places)
    expected = tuple(
        (x, y) for x in range(5) for y in range(5) if (y * y - (x**3 - x)) % 5 == 0
    )
    assert coordinates == expected
    assert len(coordinates) == 7
    assert all(place.field == field for place in result.places)
    assert all(
        place.local_parameter == ("y" if place.y == 0 else "x_minus_x0")
        for place in result.places
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_every_enumerated_place_is_accepted_by_existing_valuation():
    from jacobian.math.function_fields._models import FiniteFunctionFieldElement
    from jacobian.math.function_fields.operations import (
        function_field_hyperelliptic_affine_valuation,
    )

    field = _field()
    element = FiniteFunctionFieldElement(
        field=field, coordinates=(_rf((1,)), _rf((0,)))
    )
    for place in enumerate_hyperelliptic_affine_places(field).places:
        result = function_field_hyperelliptic_affine_valuation(place, element)
        assert result.place == place
        assert result.valuation.kind == "FINITE"
        assert result.valuation.value == 0


def test_enumeration_accepts_the_maximum_admitted_prime_characteristic():
    prime = 257

    def rational(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
        return PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(
                characteristic=prime, coefficients=coefficients
            ),
            denominator=PrimeFieldPolynomial(characteristic=prime, coefficients=(1,)),
        )

    field = FiniteFunctionField(
        characteristic=prime,
        variable="x",
        generator="y",
        defining_polynomial=(
            rational((0, 1, 0, prime - 1)),
            rational((0,)),
            rational((1,)),
        ),
    )
    result = enumerate_hyperelliptic_affine_places(field)
    assert len(result.places) <= 2 * prime
    assert len(result.places) == sum(
        (y * y - (x**3 - x)) % prime == 0 for x in range(prime) for y in range(prime)
    )


def test_tool_is_published_and_advertised_example_is_valid_json():
    operation_id = "function_field.hyperelliptic_affine_places.enumerate"
    tool = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    assert tool in BUILTIN_TOOLS
    example = tool.examples[0]
    parsed = HyperellipticAffinePlacesRequest.model_validate(example.input)
    assert enumerate_hyperelliptic_affine_places(parsed.field).places
    assert encode_strict_json(example.input)
