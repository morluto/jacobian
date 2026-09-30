"""Native and public admission agree for hyperelliptic infinity places."""

import json

import pytest
from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.function_fields import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldPlace,
    HyperellipticInfinityPlaceValuationResult,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_hyperelliptic_infinity_valuation,
)
from jacobian.math.function_fields._models import (
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
)

_VALUATION = "function_field.hyperelliptic_infinity_place.valuation.compute"


def _rf(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=coefficients),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
    )


def _field(branch: tuple[int, ...]) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=5,
        defining_polynomial=(
            _rf(tuple(-coefficient % 5 for coefficient in branch)),
            _rf((0,)),
            _rf((1,)),
        ),
    )


@pytest.mark.parametrize(
    ("branch", "expected_valuations"),
    [((0, 4, 0, 1), [0, -2, -3]), ((0, 4, 0, 0, 0, 1), [0, -2, -4, -5])],
)
def test_divisor_support_round_trips_into_native_and_public_infinity_valuation(
    branch: tuple[int, ...], expected_valuations: list[int]
) -> None:
    """L(dP) supplies unchanged generic support and the basis 1,x,...,y.

    For y^2=x^d-x, the unique infinity has v(x)=-2 and v(y)=-d.
    Both presentations must also retain the structural valuation of zero.
    """
    field = _field(branch)
    divisor = FunctionFieldDivisor(
        field=field,
        terms=(
            FunctionFieldDivisorTerm(
                place=FunctionFieldPlace(field=field, kind="INFINITE", degree=1),
                multiplicity=len(branch) - 1,
            ),
        ),
    )
    catalog = Catalog.open()
    produced = invoke_operation(
        "function_field.riemann_roch_space.compute",
        {"divisor": divisor.model_dump(mode="json")},
        catalog,
    )
    space = json.loads(produced.model_dump_json())["output"]
    place_payload = space["divisor"]["terms"][0]["place"]
    native_place = FunctionFieldPlace.model_validate_json(json.dumps(place_payload))
    descriptor = catalog.inspect(_VALUATION)
    assert descriptor is not None
    validator = Draft202012Validator(descriptor.input_schema)
    zero = FiniteFunctionFieldElement(field=field, coordinates=(_rf((0,)), _rf((0,))))
    elements = [*space["basis"], zero.model_dump(mode="json")]
    expected = [
        *({"kind": "FINITE", "value": value} for value in expected_valuations),
        {"kind": "POSITIVE_INFINITY"},
    ]
    assert len(elements) == len(expected)

    for element_payload, valuation in zip(elements, expected, strict=True):
        native = function_field_hyperelliptic_infinity_valuation(
            native_place,
            FiniteFunctionFieldElement.model_validate_json(json.dumps(element_payload)),
        )
        assert native.valuation.model_dump(mode="json") == valuation
        payload = {"place": place_payload, "element": element_payload}
        result = invoke_operation(_VALUATION, payload, catalog)
        validator.validate(payload)
        assert result.output == native.model_dump(mode="json")
        assert result.output["place"]["field"] == place_payload["field"]
        revived = HyperellipticInfinityPlaceValuationResult.model_validate_json(
            json.dumps(result.output)
        )
        assert revived == native
        # The canonical typed result place remains a public input, too.
        assert (
            invoke_operation(
                _VALUATION,
                {"place": result.output["place"], "element": element_payload},
                catalog,
            ).output
            == result.output
        )


@pytest.mark.parametrize("case", ["finite", "even_degree", "parent_mismatch"])
def test_generic_place_keeps_native_and_public_domain_rejections(case: str) -> None:
    field = _field((0, 4, 0, 1))
    if case == "even_degree":
        field = _field((1, 0, 1, 0, 1))
    place = FunctionFieldPlace(field=field, kind="INFINITE", degree=1)
    if case == "finite":
        place = FunctionFieldPlace(
            field=field,
            kind="FINITE",
            degree=1,
            prime_polynomial=PrimeFieldPolynomial(
                characteristic=5, coefficients=(0, 1)
            ),
        )
    element_field = _field((0, 4, 0, 0, 0, 1)) if case == "parent_mismatch" else field
    # Even zero must not evade place-kind, curve, or parent admission.
    element = FiniteFunctionFieldElement(
        field=element_field, coordinates=(_rf((0,)), _rf((0,)))
    )
    expected_code = {
        "finite": "function_field.infinity_place_type",
        "even_degree": "function_field.odd_hyperelliptic_infinity_required",
        "parent_mismatch": "function_field.parent_mismatch",
    }[case]

    with pytest.raises(OperationDomainValidationError) as native_error:
        function_field_hyperelliptic_infinity_valuation(place, element)
    with pytest.raises(OperationDomainValidationError) as public_error:
        invoke_operation(
            _VALUATION,
            {
                "place": place.model_dump(mode="json"),
                "element": element.model_dump(mode="json"),
            },
            Catalog.open(),
        )

    assert native_error.value.errors()[0]["type"] == expected_code
    assert public_error.value.errors()[0]["type"] == expected_code
