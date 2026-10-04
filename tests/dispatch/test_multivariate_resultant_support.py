"""Public resultant admission shares the native active-support envelope."""

import json
from math import comb

import pytest
from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.multivariate import operations
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantResult,
)
from jacobian.math.polynomials.values import RationalPolynomial

_OPERATION = "polynomial.multivariate.resultant.compute"
_VARIABLES = ("t", "y", "v", "x", "w", "u", "z", "r")


def _poly(terms: tuple[tuple[int, dict[str, int]], ...]) -> RationalPolynomial:
    coefficients = {
        tuple(powers.get(variable, 0) for variable in _VARIABLES): coefficient
        for coefficient, powers in terms
    }
    return RationalPolynomial.model_validate_json(
        json.dumps(
            {
                "variables": _VARIABLES,
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {"num": str(coefficient), "den": "1"},
                            "exponents": exponents,
                        }
                        for exponents, coefficient in sorted(
                            coefficients.items(), reverse=True
                        )
                    ]
                },
            }
        )
    )


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.open()


@pytest.mark.parametrize("power", [6, 63])
def test_public_resultant_retains_inactive_axes(catalog: Catalog, power: int) -> None:
    left = _poly(((1, {"x": 1}), (1, {"y": 1})))
    right = _poly(((1, {"x": power}),))
    native = operations.multivariate_resultant(left, right, "x")
    public = invoke_operation(
        _OPERATION,
        {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "elimination_variable": "x",
        },
        catalog,
    )
    assert public.output == native.model_dump(mode="json")
    assert public.output["resultant"]["value"] == {
        "domain": "QQ",
        "variables": [variable for variable in _VARIABLES if variable != "x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str((-1) ** power), "den": "1"},
                    "exponents": [0, power, 0, 0, 0, 0, 0],
                }
            ]
        },
    }
    decoded = MultivariateResultantResult.model_validate_json(json.dumps(public.output))
    assert operations.verify_multivariate_resultant(decoded)
    assert not operations.verify_multivariate_resultant(
        decoded.model_copy(update={"left": _poly(((1, {"x": 1}), (2, {"y": 1})))})
    )


def test_public_resultant_round_trips_exact_1024_term_boundary(
    catalog: Catalog,
) -> None:
    left = _poly(
        (
            (1, {"x": 1}),
            (-1, {"y": 1, "z": 1}),
            (-1, {"y": 1}),
            (-1, {"z": 1}),
            (-1, {}),
        )
    )
    right = _poly(((1, {"x": 31}),))
    public = invoke_operation(
        _OPERATION,
        {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "elimination_variable": "x",
        },
        catalog,
    )
    declaration = catalog.operation(_OPERATION)
    assert declaration is not None
    Draft202012Validator(
        declaration.result_type.model_json_schema(mode="serialization")
    ).validate(public.output)
    decoded = MultivariateResultantResult.model_validate_json(json.dumps(public.output))
    assert decoded == operations.multivariate_resultant(left, right, "x")
    assert decoded.resultant.kind == "POLYNOMIAL"
    assert decoded.resultant.value.variables == ("t", "y", "v", "w", "u", "z", "r")
    assert {
        term.exponents: term.coefficient.as_fraction()
        for term in decoded.resultant.value.polynomial.terms
    } == {
        (0, y, 0, 0, 0, z, 0): comb(31, y) * comb(31, z)
        for y in range(32)
        for z in range(32)
    }
    assert operations.verify_multivariate_resultant(decoded)


@pytest.mark.parametrize("family", ["axis_box", "all_active"])
def test_public_resultant_refuses_actual_output_growth_before_backend(
    catalog: Catalog, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    terms: tuple[tuple[int, dict[str, int]], ...]
    if family == "axis_box":
        terms = (
            (1, {"x": 1}),
            (-1, {"y": 1, "z": 1}),
            (-1, {"y": 1}),
            (-1, {"z": 1}),
            (-1, {}),
        )
        power = 32
    else:
        terms = tuple((1, {variable: 1}) for variable in _VARIABLES)
        power = 7
    left = _poly(terms)
    right = _poly(((1, {"x": power}),))

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("public resultant admission must precede the backend")

    monkeypatch.setattr(operations, "_sylvester_resultant_value", unexpected_backend)
    with pytest.raises(OperationDomainValidationError) as public:
        invoke_operation(
            _OPERATION,
            {
                "left": left.model_dump(mode="json"),
                "right": right.model_dump(mode="json"),
                "elimination_variable": "x",
            },
            catalog,
        )
    assert (
        public.value.errors()[0]["type"]
        == "polynomial.multivariate_resultant.support_budget"
    )
    with pytest.raises(OperationDomainValidationError) as native:
        operations.multivariate_resultant(left, right, "x")
    assert (
        native.value.errors()[0]["type"]
        == "polynomial.multivariate_resultant.support_budget"
    )
    assert public.value.errors() == native.value.errors()
