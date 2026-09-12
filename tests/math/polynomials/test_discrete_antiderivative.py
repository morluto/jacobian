"""Selected-variable rational discrete antiderivative tests."""

from fractions import Fraction
from math import prod

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._discrete_antiderivative import (
    RationalDiscreteAntiderivativeRequest,
    RationalDiscreteAntiderivativeResult,
    rational_discrete_antiderivative,
)
from jacobian.math.polynomials._discrete_antiderivative_tools import (
    RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(terms: tuple[tuple[int, tuple[int, ...]], ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("k", "N"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _evaluate(polynomial: RationalPolynomial, values: dict[str, int]) -> Fraction:
    return sum(
        (
            term.coefficient.as_fraction()
            * prod(
                Fraction(values[variable]) ** exponent
                for variable, exponent in zip(
                    polynomial.variables, term.exponents, strict=True
                )
            )
            for term in polynomial.polynomial.terms
        ),
        Fraction(),
    )


def test_retained_k_squared_n_minus_k_squared_fixture() -> None:
    source = _polynomial(((1, (4, 0)), (-2, (3, 1)), (1, (2, 2))))
    result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
    )
    assert result.reconstructed_difference == source
    assert (
        RationalDiscreteAntiderivativeResult.model_validate_json(
            result.model_dump_json()
        )
        == result
    )
    assert all(
        term.exponents[0] >= 1 for term in result.antiderivative.polynomial.terms
    )
    for n in range(1, 9):
        endpoint_sum = _evaluate(
            result.antiderivative,
            {"k": n, "N": n},
        ) - _evaluate(result.antiderivative, {"k": 1, "N": n})
        assert endpoint_sum == Fraction(n**5 - n, 30)


@pytest.mark.parametrize("degree", range(6))
def test_monomials_reconstruct_exactly_through_degree_five(degree: int) -> None:
    source = _polynomial(((1, (degree, 0)),))
    result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
    )
    assert result.reconstructed_difference == source
    assert result.antiderivative.variables == source.variables
    assert all(
        term.exponents[0] >= 1 for term in result.antiderivative.polynomial.terms
    )


def test_zero_and_constant_sources_preserve_axes() -> None:
    zero = _polynomial(())
    zero_result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=zero, variable="k")
    )
    assert zero_result.antiderivative.variables == ("k", "N")
    assert zero_result.antiderivative.polynomial.terms == ()
    constant = _polynomial(((7, (0, 0)),))
    constant_result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=constant, variable="k")
    )
    assert constant_result.reconstructed_difference == constant
    assert tuple(
        (term.coefficient.as_fraction(), term.exponents)
        for term in constant_result.antiderivative.polynomial.terms
    ) == ((Fraction(7), (1, 0)),)


def test_other_variable_is_a_coefficient_parameter() -> None:
    source = _polynomial(((1, (1, 1)),))
    result = rational_discrete_antiderivative(
        RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
    )
    coefficients = {
        term.exponents: term.coefficient.as_fraction()
        for term in result.antiderivative.polynomial.terms
    }
    assert coefficients == {(2, 1): Fraction(1, 2), (1, 1): Fraction(-1, 2)}


def test_denominator_growth_is_rejected_before_result_construction() -> None:
    denominator = 5 * 10**32_767 + 1
    source = RationalPolynomial(
        variables=("k",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=denominator),
                    exponents=(1,),
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        rational_discrete_antiderivative(
            RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
        )


def test_quadratic_triangular_work_is_rejected_before_expansion() -> None:
    source = _polynomial(((1, (1_024, 0)),))
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        rational_discrete_antiderivative(
            RationalDiscreteAntiderivativeRequest(polynomial=source, variable="k")
        )


def test_request_type_and_selected_axis_are_domain_errors() -> None:
    with pytest.raises(OperationDomainValidationError, match="request must be"):
        rational_discrete_antiderivative(object())  # type: ignore[arg-type]
    source = _polynomial(((1, (1, 0)),))
    with pytest.raises(OperationDomainValidationError, match="selected variable"):
        rational_discrete_antiderivative(
            RationalDiscreteAntiderivativeRequest(polynomial=source, variable="z")
        )


def test_schema_and_example_explain_selected_axis_contract() -> None:
    schema = RationalDiscreteAntiderivativeRequest.model_json_schema()
    assert "variable" in schema["properties"]
    assert RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples
    assert "selected variable" in (
        RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples[0].description
    )


def test_catalog_invocation_returns_the_declared_typed_result() -> None:
    catalog = Catalog.open()
    invocation = RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples[0]
    result = invoke_operation(
        RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.operation_id,
        invocation.input,
        catalog,
    )
    assert set(result.output) == {
        "source",
        "antiderivative",
        "reconstructed_difference",
    }
