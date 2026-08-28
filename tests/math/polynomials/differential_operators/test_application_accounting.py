"""Observed-output evidence for differential-operator application admission."""

from __future__ import annotations

from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.differential_operators._bounds import (
    validate_application_envelope,
)
from jacobian.math.polynomials.differential_operators._models import (
    DifferentialOperatorApplyRequest,
)
from jacobian.math.polynomials.differential_operators._tools import (
    compute_differential_operator_application,
)
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
    DifferentialOperatorTerm,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _one() -> CanonicalRational:
    return CanonicalRational(num="1", den="1")


def _request() -> DifferentialOperatorApplyRequest:
    source = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(coefficient=_one(), exponents=(2,)),
                RationalPolynomialTerm(coefficient=_one(), exponents=(1,)),
                RationalPolynomialTerm(coefficient=_one(), exponents=(0,)),
            )
        ),
    )
    operator = ConstantCoefficientDifferentialOperator(
        variables=("x",),
        terms=(
            DifferentialOperatorTerm(coefficient=_one(), orders=(1,)),
            DifferentialOperatorTerm(coefficient=_one(), orders=(0,)),
        ),
    )
    return DifferentialOperatorApplyRequest(polynomial=source, operator=operator)


def test_application_charges_every_materialized_output_term() -> None:
    request = _request()
    envelope = validate_application_envelope(
        request.polynomial,
        request.operator,
        request.iterations,
        request.expected,
    )
    result = compute_differential_operator_application(request)

    assert_charged_work_parity(
        charged={"output_term": envelope.candidate_output_terms},
        executed={"output_term": len(result.output.polynomial.terms)},
    )
    assert result.output.polynomial.terms
