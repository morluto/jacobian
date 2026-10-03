"""Catalog schema check for the restored residue-image bounds."""

from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory.modular_polynomials import (
    ModularPolynomialTerm,
)


def _request(*, exponent: int = 1) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient=3, exponents=(exponent,)),),
    )


def _request_with_coefficient(
    coefficient: int,
) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient=coefficient, exponents=(1,)),),
    )


def _request_payload(coefficient: str) -> dict[str, object]:
    return {
        "modulus": 5,
        "variables": [{"name": "x", "residues": [0, 1]}],
        "terms": [{"coefficient": coefficient, "exponents": [1]}],
    }


def test_both_residue_image_operations_advertise_the_restored_bounds() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    residue_image_tools = [
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id.startswith("modular.polynomial_residue_image.")
    ]
    assert len(residue_image_tools) == 2

    for tool in residue_image_tools:
        exponents_schema = tool.result_type.model_json_schema()["properties"][
            "normalized_terms"
        ]["items"]["properties"]["exponents"]
        assert exponents_schema["maxItems"] == 6
        assert exponents_schema["items"]["minimum"] == 0
        assert exponents_schema["items"]["maximum"] == 32
