"""Defining identities for bounded ideal-membership certificates."""

from __future__ import annotations

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials.ideals._models import (
    IdealMembershipCertificateRequest,
    IdealMembershipCertificateResult,
)
from jacobian.math.polynomials.ideals.operations import ideal_membership_certificate
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...], terms: tuple[tuple[int, tuple[int, ...]], ...]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=str(coefficient), den="1"),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _request(
    generators: tuple[RationalPolynomial, ...],
    target: RationalPolynomial,
    degree: int,
) -> IdealMembershipCertificateRequest:
    return IdealMembershipCertificateRequest(
        ideal=RationalPolynomialIdeal(
            variables=target.variables, generators=generators
        ),
        polynomial=target,
        cofactor_degree_bound=degree,
    )


def test_certificate_reconstructs_the_source_generator_identity() -> None:
    variables = ("x", "y")
    request = _request(
        (
            _polynomial(variables, ((1, (1, 0)), (-1, (0, 1)))),
            _polynomial(variables, ((1, (1, 0)), (1, (0, 1)))),
        ),
        _polynomial(variables, ((1, (2, 0)), (-1, (0, 2)))),
        1,
    )

    result = ideal_membership_certificate(
        request.ideal, request.polynomial, request.cofactor_degree_bound
    )

    assert result.status == "CERTIFICATE"
    assert result.multiplier is not None
    assert result.cofactors is not None
    reconstructed = sum(
        rational_polynomial_to_sympy(cofactor).as_expr()
        * rational_polynomial_to_sympy(generator).as_expr()
        for cofactor, generator in zip(
            result.cofactors, request.ideal.generators, strict=True
        )
    )
    target = rational_polynomial_to_sympy(request.polynomial).as_expr()
    assert sympy.expand(reconstructed - int(result.multiplier) * target) == 0


def test_certificate_clears_rational_cofactor_denominators_primitively() -> None:
    variables = ("x",)
    request = _request(
        (_polynomial(variables, ((2, (1,)),)),),
        _polynomial(variables, ((1, (1,)),)),
        0,
    )

    result = ideal_membership_certificate(
        request.ideal, request.polynomial, request.cofactor_degree_bound
    )

    assert result.status == "CERTIFICATE"
    assert result.multiplier == "2"
    assert result.cofactors is not None
    assert result.cofactors[0] == _polynomial(variables, ((1, (0,)),))


def test_negative_result_is_limited_to_the_declared_cofactor_degree() -> None:
    variables = ("x",)
    request = _request(
        (_polynomial(variables, ((1, (2,)),)),),
        _polynomial(variables, ((1, (1,)),)),
        3,
    )

    result = ideal_membership_certificate(
        request.ideal, request.polynomial, request.cofactor_degree_bound
    )

    assert result.status == "NO_CERTIFICATE_WITHIN_BOUND"
    assert result.multiplier is None
    assert result.cofactors is None
    assert (
        IdealMembershipCertificateResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_certificate_search_rejects_expansion_before_enumeration() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    zero = _polynomial(variables, ())
    request = _request((zero,), zero, 16)

    with pytest.raises(OperationDomainValidationError, match="column"):
        ideal_membership_certificate(
            request.ideal, request.polynomial, request.cofactor_degree_bound
        )
