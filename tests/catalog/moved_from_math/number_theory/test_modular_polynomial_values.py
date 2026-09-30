"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/test_modular_polynomial_values.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

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
