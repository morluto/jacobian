"""Exact Gaussian polynomial moment operation contracts."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
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


def _operation():
    return next(
        operation
        for operation in finite_probability_operations()
        if operation.operation_id == "probability.gaussian_polynomial.moment.compute"
    )


def _term(exponents: tuple[int, ...]) -> GaussianPolynomialTerm:
    return GaussianPolynomialTerm(
        coefficient=ExactComplexRational(
            real={"num": "1", "den": "1"},
            imaginary={"num": "0", "den": "1"},
        ),
        exponents=exponents,
    )


def test_gaussian_moment_example_round_trips_as_trusted_result() -> None:
    operation = _operation()
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)

    assert result.moment.as_fractions() == (2, 0)
    assert GaussianPolynomialMomentResult.model_validate(result.model_dump()) == result


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
