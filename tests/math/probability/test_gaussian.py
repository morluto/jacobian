"""Exact Gaussian polynomial moment operation contracts."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.probability._gaussian import (
    MAX_GAUSSIAN_EXPANSION_PATHS,
    ExactComplexRational,
    GaussianPolynomial,
    GaussianPolynomialMomentResult,
    GaussianPolynomialTerm,
)
from jacobian.math.probability._gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)
from jacobian.math.probability._tools import finite_probability_operations


def _operation() -> MathTool[
    CanonicalGaussianPolynomialMomentRequest, GaussianPolynomialMomentResult
]:
    return next(
        operation
        for operation in finite_probability_operations()
        if operation.operation_id == "probability.gaussian_polynomial.moment.compute"
    )


def _term(exponents: tuple[int, ...]) -> GaussianPolynomialTerm:
    return GaussianPolynomialTerm(
        coefficient=ExactComplexRational(
            real=CanonicalRational(num=1, den=1),
            imaginary=CanonicalRational(num=0, den=1),
        ),
        exponents=exponents,
    )


def test_gaussian_moment_example_round_trips_as_trusted_result() -> None:
    operation = _operation()
    request = CanonicalGaussianPolynomialMomentRequest(
        polynomial=GaussianPolynomial(
            variable_count=2,
            terms=(_term((0, 1)), _term((1, 0))),
        ),
        order=2,
    )
    result = operation.run(request)

    assert result.moment.as_fractions() == (Fraction(2), Fraction(0))
    assert (
        GaussianPolynomialMomentResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_expansion_bound_is_admitted_at_operation_time() -> None:
    polynomial = GaussianPolynomial(
        variable_count=2,
        terms=tuple(
            _term(exponents)
            for exponents in sorted((*((index, 0) for index in range(9)), (0, 1)))
        ),
    )
    request = CanonicalGaussianPolynomialMomentRequest(polynomial=polynomial, order=6)

    assert len(polynomial.terms) ** request.order > MAX_GAUSSIAN_EXPANSION_PATHS
    with pytest.raises(OperationDomainValidationError, match="path expansion bound"):
        _operation().run(request)
