"""Typed formal modular-polynomial identity operation."""

from jacobian.catalog._examples import example
from jacobian.math.modular_polynomials import (
    ModularPolynomialIdentityRequest,
    ModularPolynomialIdentityValue,
    _modular_polynomial_identity_request,
)
from jacobian.math.number_theory._support import number_theory_operation

MODULAR_IDENTITY_OPERATIONS = (
    number_theory_operation(
        "modular.polynomial_identity.compute",
        "Compare modular polynomial coefficients",
        (
            "Canonicalize two sparse integer polynomials and compare their formal "
            "coefficients modulo m. This is polynomial-ring identity, not equality "
            "of induced functions on residue assignments."
        ),
        ModularPolynomialIdentityRequest,
        ModularPolynomialIdentityValue,
        _modular_polynomial_identity_request,
        "number-theory",
        "modular",
        "polynomial",
        "identity",
        "coefficientwise",
        examples=(
            example(
                "coefficientwise_identity_mod_4",
                "Compare two formal polynomial coefficients modulo 4.",
                {
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
