"""Typed formal modular-polynomial identity operation."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.modular_polynomials import (
    ModularPolynomialIdentityRequest,
    ModularPolynomialIdentityValue,
    modular_polynomial_identity,
)


def _run_modular_polynomial_identity(
    request: ModularPolynomialIdentityRequest,
) -> ModularPolynomialIdentityValue:
    return modular_polynomial_identity(
        request.modulus, request.variables, request.left, request.right
    )


MODULAR_IDENTITY_OPERATIONS = (
    MathTool(
        operation_id="modular.polynomial_identity.compute",
        title="Compare modular polynomial coefficients",
        description=(
            "Canonicalize two sparse integer polynomials and compare their formal "
            "coefficients modulo m. This is polynomial-ring identity, not equality "
            "of induced functions on residue assignments."
        ),
        request_type=ModularPolynomialIdentityRequest,
        result_type=ModularPolynomialIdentityValue,
        run=_run_modular_polynomial_identity,
        tags=("number-theory", "modular", "polynomial", "identity", "coefficientwise"),
        examples=(
            OperationExample(
                name="coefficientwise_identity_mod_4",
                description="Compare two formal polynomial coefficients modulo 4.",
                input={
                    "modulus": 4,
                    "variables": ["z"],
                    "left": [{"coefficient": "9", "exponents": [6]}],
                    "right": [{"coefficient": "-7", "exponents": [6]}],
                },
            ),
        ),
    ),
)

__all__ = ["MODULAR_IDENTITY_OPERATIONS"]
