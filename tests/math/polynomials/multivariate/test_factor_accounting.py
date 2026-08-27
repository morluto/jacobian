"""Focused accounting evidence for multivariate factorization."""

from __future__ import annotations

from fractions import Fraction
from unittest.mock import patch

from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.multivariate import _factor_backend
from jacobian.math.polynomials.multivariate._factor_models import (
    MultivariateFactorRequest,
)
from jacobian.math.polynomials.multivariate._operations import multivariate_factor
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _factorable_polynomial() -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(1)),
                    exponents=(2, 1),
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(-1)),
                    exponents=(1, 0),
                ),
            )
        ),
    )


def test_factorization_charges_its_single_backend_job() -> None:
    with patch.object(
        _factor_backend,
        "run_bounded_factorization",
        wraps=_factor_backend.run_bounded_factorization,
    ) as run:
        result = multivariate_factor(
            MultivariateFactorRequest(polynomial=_factorable_polynomial())
        )

    assert result.factors
    assert_charged_work_parity(
        charged={"worker": 1}, executed={"worker": run.call_count}
    )
