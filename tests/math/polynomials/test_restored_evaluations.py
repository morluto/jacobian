import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._elementary import INTEGER_POLYNOMIAL_OPERATIONS
from jacobian.math.polynomials._elementary_kernel import (
    integer_polynomial_evaluate,
    rational_polynomial_evaluate,
)
from jacobian.math.polynomials._models import (
    IntegerPolynomial,
    IntegerPolynomialEvaluationRequest,
    RationalPolynomialEvaluationRequest,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def test_integer_polynomial_evaluation_is_published_and_exact() -> None:
    request = IntegerPolynomialEvaluationRequest(
        polynomial=IntegerPolynomial(coefficients=("2", "-3", "1")),
        point="4",
    )

    assert integer_polynomial_evaluate(request.polynomial, request.point).value == "21"
    assert any(
        operation.operation_id == "polynomial.integer.compute.evaluate"
        for operation in INTEGER_POLYNOMIAL_OPERATIONS
    )


def test_rational_polynomial_evaluation_is_exact() -> None:
    polynomial = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"), exponents=(2,)
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"), exponents=(0,)
                ),
            )
        ),
    )
    request = RationalPolynomialEvaluationRequest(
        polynomial=polynomial,
        point=CanonicalRational(num="2", den="1"),
    )
    result = rational_polynomial_evaluate(request.polynomial, request.point)

    assert result.value == CanonicalRational(num="5", den="1")


def test_rational_polynomial_evaluation_rejects_oversized_exact_result() -> None:
    polynomial = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num="1", den="1"),
                    exponents=(64,),
                ),
            )
        ),
    )
    request = RationalPolynomialEvaluationRequest(
        polynomial=polynomial,
        point=CanonicalRational(
            num="1" + "0" * (MAX_CANONICAL_RATIONAL_DIGITS - 1),
            den="1",
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        rational_polynomial_evaluate(request.polynomial, request.point)

    assert exc_info.value.errors()[0]["type"] == (
        "polynomial.evaluation_result_exceeds_component_bound"
    )
