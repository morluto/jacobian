"""Canonical Gaussian request parsing at the probability-domain boundary."""

import pytest
from pydantic import ValidationError

from jacobian.contracts.probability import GaussianPolynomial
from jacobian.domains.probability.bundle import build_finite_probability_bundle
from jacobian.domains.probability.gaussian_inputs import (
    CanonicalGaussianPolynomialMomentRequest,
)


def _q(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def _term(
    exponents: list[int], *, real: int = 0, imaginary: int = 0
) -> dict[str, object]:
    return {
        "coefficient": {"real": _q(real), "imaginary": _q(imaginary)},
        "exponents": exponents,
    }


def test_request_canonicalizes_unordered_duplicates_and_zero_terms() -> None:
    request = CanonicalGaussianPolynomialMomentRequest.model_validate(
        {
            "polynomial": {
                "variable_count": 2,
                "terms": [
                    _term([1, 0], real=1, imaginary=1),
                    _term([0, 1], real=4),
                    _term([1, 0], real=2, imaginary=-1),
                    _term([0, 0]),
                ],
            },
            "order": 2,
        }
    )
    assert tuple(term.exponents for term in request.polynomial.terms) == (
        (0, 1),
        (1, 0),
    )
    assert tuple(
        term.coefficient.as_fractions() for term in request.polynomial.terms
    ) == ((4, 0), (3, 0))


def test_raw_term_limit_is_enforced_before_duplicates_are_combined() -> None:
    with pytest.raises(ValidationError, match="16"):
        CanonicalGaussianPolynomialMomentRequest.model_validate(
            {
                "polynomial": {
                    "variable_count": 1,
                    "terms": [_term([0], real=1) for _ in range(17)],
                },
                "order": 1,
            }
        )


def test_request_rejects_a_polynomial_that_canonicalizes_to_zero() -> None:
    with pytest.raises(ValidationError, match="removed every zero term"):
        CanonicalGaussianPolynomialMomentRequest.model_validate(
            {
                "polynomial": {
                    "variable_count": 1,
                    "terms": [_term([1], real=3), _term([1], real=-3)],
                },
                "order": 1,
            }
        )


def test_direct_gaussian_polynomial_value_remains_strict() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        GaussianPolynomial.model_validate(
            {
                "variable_count": 2,
                "terms": [_term([1, 0], real=1), _term([0, 1], real=1)],
            }
        )


def test_producer_and_checker_share_the_canonical_request_owner() -> None:
    bundle = build_finite_probability_bundle()
    producer = next(
        operation
        for operation in bundle.capabilities
        if operation.spec.operation_id
        == "probability.gaussian_polynomial.moment.compute"
    )
    checker = next(
        declaration
        for declaration in bundle.checker_declarations
        if declaration.capability_id == "probability.gaussian_polynomial.moment.compute"
    )
    assert producer.spec.request_type is CanonicalGaussianPolynomialMomentRequest
    assert checker.request_model is CanonicalGaussianPolynomialMomentRequest
