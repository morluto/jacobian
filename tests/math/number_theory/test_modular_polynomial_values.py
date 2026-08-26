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


def _request_with_coefficient(
    coefficient: str,
) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient=coefficient, exponents=(1,)),),
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


def test_published_term_schema_matches_residue_image_admission() -> None:
    term_schema = ModularPolynomialResidueImageRequest.model_json_schema()[
        "properties"
    ]["terms"]["items"]

    assert term_schema["properties"]["coefficient"]["maxLength"] == 256
    assert term_schema["properties"]["exponents"]["maxItems"] == 6
    assert term_schema["properties"]["exponents"]["items"]["maximum"] == 32

    shared_schema = ModularPolynomialTerm.model_json_schema()
    assert shared_schema["properties"]["coefficient"]["maxLength"] == 257
    assert shared_schema["properties"]["exponents"]["maxItems"] == 20


def test_coefficient_boundary_follows_the_advertised_envelope() -> None:
    admitted = _request_with_coefficient("-" + "9" * 255)
    assert int(admitted.terms[0].coefficient) < 0

    with expect_validation("number_theory.term_outside_residue_image_admission"):
        _request_with_coefficient("-" + "9" * 256)


def test_shared_term_type_retains_its_wider_envelope_elsewhere() -> None:
    widest = "-" + "9" * 256
    identity = modular_polynomial_identity(
        ModularPolynomialIdentityRequest(
            modulus=5,
            variables=("x",),
            left=(ModularPolynomialTerm(coefficient=widest, exponents=(1,)),),
        )
    )

    assert identity.normalized_left[0].coefficient == int(widest) % 5
