from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._elementary import INTEGER_POLYNOMIAL_OPERATIONS
from jacobian.math.polynomials._elementary_operations import (
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

    assert integer_polynomial_evaluate(request).value == "21"
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
    result = rational_polynomial_evaluate(
        RationalPolynomialEvaluationRequest(
            polynomial=polynomial,
            point=CanonicalRational(num="2", den="1"),
        )
    )

    assert result.value == CanonicalRational(num="5", den="1")
