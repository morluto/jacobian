"""Composition checks for sparse modular-polynomial values."""

from tests.math.number_theory._validation import expect_validation

from jacobian.math.modular_polynomials import (
    ModularPolynomialIdentityRequest,
    ModularPolynomialTerm,
    NormalizedModularPolynomialTerm,
    modular_polynomial_identity,
)
from jacobian.math.number_theory._models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory._modular_operations import (
    compute_modular_polynomial_residue_image,
)


def _request(*, exponent: int = 1) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient="3", exponents=(exponent,)),),
    )


def test_residue_image_consumes_and_produces_the_canonical_term_types() -> None:
    request = _request()

    assert type(request.terms[0]) is ModularPolynomialTerm
    assert request.model_dump(mode="json")["terms"] == [
        {"coefficient": "3", "exponents": [1]}
    ]

    identity = modular_polynomial_identity(
        ModularPolynomialIdentityRequest(
            modulus=5,
            variables=("x",),
            left=request.terms,
        )
    )
    result = compute_modular_polynomial_residue_image(request)

    assert type(result.normalized_terms[0]) is NormalizedModularPolynomialTerm
    assert identity.normalized_left == result.normalized_terms
    assert result.model_dump(mode="json")["normalized_terms"] == [
        {"coefficient": 3, "exponents": [1]}
    ]


def test_residue_image_keeps_its_narrower_exponent_admission() -> None:
    with expect_validation("number_theory.term_outside_residue_image_admission"):
        _request(exponent=33)
