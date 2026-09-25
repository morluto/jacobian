"""Exact coefficient embedding into rational cyclotomic polynomial values."""

import json

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.polynomials.cyclotomic_coefficients._models import (
    CyclotomicPolynomial,
    CyclotomicPolynomialTerm,
)
from jacobian.math.polynomials.cyclotomic_coefficients._tools import TOOLS
from jacobian.math.polynomials.cyclotomic_coefficients.operations import (
    embed_rational_polynomial,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def rational_term(coefficient: int, *exponents: int) -> RationalPolynomialTerm:
    return RationalPolynomialTerm(
        coefficient=CanonicalRational(num=coefficient, den=1),
        exponents=exponents,
    )


def test_embedding_preserves_sparse_polynomial_context_and_exact_coefficients() -> None:
    source = RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                rational_term(2, 1, 0),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=-3, den=2),
                    exponents=(0, 2),
                ),
            )
        ),
    )
    field = RationalCyclotomicField(order=5)

    result = embed_rational_polynomial(source, field)

    assert result.domain == "QQ_CYCLOTOMIC_POLYNOMIAL"
    assert result.field == field
    assert result.variables == source.variables
    assert tuple(term.exponents for term in result.terms) == tuple(
        term.exponents for term in source.polynomial.terms
    )
    assert tuple(term.coefficient.coefficients_ascending for term in result.terms) == (
        (
            CanonicalRational(num=2, den=1),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
        ),
        (
            CanonicalRational(num=-3, den=2),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    assert CyclotomicPolynomial.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("field_order", [1, 5])
def test_zero_polynomial_retains_axes_and_field(field_order: int) -> None:
    source = RationalPolynomial(
        variables=("x", "y"), polynomial=SparseRationalPolynomial(terms=())
    )
    field = RationalCyclotomicField(order=field_order)

    result = embed_rational_polynomial(source, field)

    assert result.terms == ()
    assert result.variables == ("x", "y")
    assert result.field == field


def test_constant_polynomial_and_order_one_field() -> None:
    source = RationalPolynomial(
        variables=(), polynomial=SparseRationalPolynomial(terms=(rational_term(7),))
    )

    result = embed_rational_polynomial(source, RationalCyclotomicField(order=1))

    assert result.terms[0].exponents == ()
    assert result.terms[0].coefficient.coefficients_ascending == (
        CanonicalRational(num=7, den=1),
    )


def test_field_mismatch_and_zero_coefficient_are_rejected() -> None:
    field = RationalCyclotomicField(order=5)
    coefficient = RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=0, den=1) for _ in range(field.degree)
        ),
    )
    with pytest.raises(ValidationError, match="zero terms must be omitted"):
        CyclotomicPolynomialTerm(coefficient=coefficient, exponents=(0,))
    nonzero = RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            CanonicalRational(num=1, den=1),
            *(CanonicalRational(num=0, den=1) for _ in range(field.degree - 1)),
        ),
    )
    with pytest.raises(
        ValidationError, match="every coefficient must use the polynomial field"
    ):
        CyclotomicPolynomial(
            field=RationalCyclotomicField(order=1),
            variables=("x",),
            terms=(CyclotomicPolynomialTerm(coefficient=nonzero, exponents=(0,)),),
        )


def test_public_manifest_declares_composable_typed_coefficient_map() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "polynomial.cyclotomic_coefficient.embed.compute",
    )
    tool = TOOLS[0]
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert isinstance(tool.run(request), CyclotomicPolynomial)


def test_embedding_admits_coordinate_growth_before_expansion() -> None:
    field = RationalCyclotomicField(order=128)
    maximum_terms = tuple(
        rational_term(index + 1, index) for index in range(255, -1, -1)
    )
    at_bound = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(terms=maximum_terms),
    )
    accepted = embed_rational_polynomial(at_bound, field)
    assert len(accepted.terms) * field.degree == 16_384

    source = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(rational_term(257, 256), *maximum_terms)
        ),
    )

    with pytest.raises(OperationResourceAdmissionError, match="coordinates"):
        embed_rational_polynomial(source, field)


def test_raw_nested_parent_coordinates_are_admitted_before_model_construction() -> None:
    raw = {
        "field": {"order": 1},
        "variables": ["x"],
        "terms": [
            {
                "coefficient": {
                    "field": {"order": 128},
                    "coefficients_ascending": [{"num": 1, "den": 1}] * 65,
                },
                "exponents": [0],
            }
        ]
        * 253,
    }
    with pytest.raises(ValidationError, match="coefficient coordinates"):
        CyclotomicPolynomial.model_validate(raw)


def test_embedding_native_boundary_rejects_untyped_and_forged_values() -> None:
    with pytest.raises(OperationDomainValidationError):
        embed_rational_polynomial(object(), object())  # type: ignore[arg-type]
    forged = object.__new__(RationalPolynomial)
    with pytest.raises(OperationDomainValidationError):
        embed_rational_polynomial(forged, RationalCyclotomicField(order=1))


def test_embedding_rejects_coefficients_that_do_not_fit_exact_scalar_carrier() -> None:
    source = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=10**256, den=1),
                    exponents=(1,),
                ),
            )
        ),
    )

    with pytest.raises(OperationResourceAdmissionError, match="height"):
        embed_rational_polynomial(source, RationalCyclotomicField(order=5))
