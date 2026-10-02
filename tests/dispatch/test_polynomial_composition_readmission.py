"""Published polynomial producers compose without caller-side canonical repair."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._models import PolynomialFactorizationResult
from jacobian.math.polynomials.maps._models import CompositionResult
from jacobian.math.polynomials.operations import verify_polynomial_factorization


def _monomial(
    degree: int, coefficient: str = "1", variable: str = "x"
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": [variable],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": coefficient, "den": "1"}, "exponents": [degree]}
            ]
        },
    }


@pytest.mark.parametrize("coefficient_growth", (False, True))
@pytest.mark.parametrize("identity_side", ("outer", "inner"))
def test_public_composition_reuses_actual_output_unchanged(
    coefficient_growth: bool, identity_side: str
) -> None:
    catalog = Catalog.open()
    operation = "polynomial.map.compose"
    first = invoke_operation(
        operation,
        {
            "outer": _monomial(64),
            "inner": _monomial(1, "1" + "0" * 127)
            if coefficient_growth
            else _monomial(2),
            "outer_variable": "x",
            "inner_variable": "x",
        },
        catalog,
    ).output
    expected = _monomial(64, "1" + "0" * 8128) if coefficient_growth else _monomial(128)
    assert first == {"polynomial": expected}
    assert (
        CompositionResult.model_validate_json(json.dumps(first)).model_dump(mode="json")
        == first
    )
    payload = {
        "outer": _monomial(1) if identity_side == "outer" else first["polynomial"],
        "inner": first["polynomial"] if identity_side == "outer" else _monomial(1),
        "outer_variable": "x",
        "inner_variable": "x",
    }
    second = invoke_operation(operation, payload, catalog).output
    assert second == first
    assert (
        CompositionResult.model_validate_json(json.dumps(second)).model_dump(
            mode="json"
        )
        == second
    )


def test_public_identity_composition_preserves_explicit_renaming() -> None:
    output = invoke_operation(
        "polynomial.map.compose",
        {
            "outer": _monomial(128, "3", "u"),
            "inner": _monomial(1),
            "outer_variable": "u",
            "inner_variable": "x",
        },
        Catalog.open(),
    ).output
    assert output == {"polynomial": _monomial(128, "3")}


def test_public_nonidentity_composition_keeps_typed_refusal() -> None:
    with pytest.raises(OperationDomainValidationError):
        invoke_operation(
            "polynomial.map.compose",
            {
                "outer": _monomial(64),
                "inner": _monomial(3),
                "outer_variable": "x",
                "inner_variable": "x",
            },
            Catalog.open(),
        )


def test_public_factorization_complete_output_decodes_and_verifies() -> None:
    coefficients = [1]
    for root in range(1, 66):
        product = [0] * (len(coefficients) + 1)
        for degree, coefficient in enumerate(coefficients):
            product[degree] -= root * coefficient
            product[degree + 1] += coefficient
        coefficients = product
    source = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": str(value), "den": "1"}, "exponents": [degree]}
                for degree, value in reversed(tuple(enumerate(coefficients)))
            ]
        },
    }
    output = invoke_operation(
        "polynomial.factor.compute", {"polynomial": source}, Catalog.open()
    ).output
    assert len(output["factors"]) == 65
    assert output["polynomial"] == output["reconstructed"] == source
    claim = PolynomialFactorizationResult.model_validate_json(json.dumps(output))
    assert claim.model_dump(mode="json") == output
    assert verify_polynomial_factorization(claim)
