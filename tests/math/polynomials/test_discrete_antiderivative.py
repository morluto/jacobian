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
        source,
        "k",
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
        source,
        "k",
    )
    assert result.reconstructed_difference == source
    assert result.antiderivative.variables == source.variables
    assert all(
        term.exponents[0] >= 1 for term in result.antiderivative.polynomial.terms
    )


def test_zero_and_constant_sources_preserve_axes() -> None:
    zero = _polynomial(())
    zero_result = rational_discrete_antiderivative(
        zero,
        "k",
    )
    assert zero_result.antiderivative.variables == ("k", "N")
    assert zero_result.antiderivative.polynomial.terms == ()
    constant = _polynomial(((7, (0, 0)),))
    constant_result = rational_discrete_antiderivative(
        constant,
        "k",
    )
    assert constant_result.reconstructed_difference == constant
    assert tuple(
        (term.coefficient.as_fraction(), term.exponents)
        for term in constant_result.antiderivative.polynomial.terms
    ) == ((Fraction(7), (1, 0)),)


def test_other_variable_is_a_coefficient_parameter() -> None:
    source = _polynomial(((1, (1, 1)),))
    result = rational_discrete_antiderivative(
        source,
        "k",
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
            source,
            "k",
        )


def test_linear_coefficient_at_the_digit_limit_has_an_exact_inverse() -> None:
    coefficient = 10**32_767 + 1
    source = RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=(1,),
                ),
            )
        ),
    )
    result = rational_discrete_antiderivative(source, "x")
    terms = {
        term.exponents: term.coefficient.as_fraction()
        for term in result.antiderivative.polynomial.terms
    }
    expected = Fraction(coefficient, 2)
    assert terms == {(2,): expected, (1,): -expected}
    assert result.reconstructed_difference == source


def test_quadratic_triangular_work_is_rejected_before_expansion() -> None:
    source = _polynomial(((1, (1_024, 0)),))
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        rational_discrete_antiderivative(
            source,
            "k",
        )


def test_request_type_and_selected_axis_are_domain_errors() -> None:
    with pytest.raises(OperationDomainValidationError, match="RationalPolynomial"):
        rational_discrete_antiderivative(object(), "k")  # type: ignore[arg-type]
    source = _polynomial(((1, (1, 0)),))
    with pytest.raises(OperationDomainValidationError, match="selected variable"):
        rational_discrete_antiderivative(
            source,
            "z",
        )


@pytest.mark.parametrize(
    "forged_polynomial",
    (
        RationalPolynomial.model_construct(
            variables=("k",),
            polynomial=object(),
        ),
        RationalPolynomial.model_construct(
            variables=("k",),
            polynomial=SparseRationalPolynomial.model_construct(terms=(object(),)),
        ),
        RationalPolynomial.model_construct(
            variables=("k",),
            polynomial=SparseRationalPolynomial.model_construct(
                terms=(RationalPolynomialTerm.model_construct(),)
            ),
        ),
    ),
)
def test_native_boundary_rejects_malformed_nested_polynomial_values(
    forged_polynomial: RationalPolynomial,
) -> None:
    with pytest.raises(OperationDomainValidationError):
        rational_discrete_antiderivative(forged_polynomial, "k")


def test_schema_and_example_explain_selected_axis_contract() -> None:
    schema = RationalDiscreteAntiderivativeRequest.model_json_schema()
    assert "variable" in schema["properties"]
    assert "1000000" in schema["properties"]["polynomial"]["description"]
    assert "1000000" in RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.description
    assert RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples
    assert "selected variable" in (
        RATIONAL_DISCRETE_ANTIDERIVATIVE_OPERATION.examples[0].description
    )


def test_admitted_solve_honors_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event

    from jacobian._execution import (
        OperationExecutionCancelledError,
        request_cancellation,
        request_checkpoint,
    )
    from jacobian.math.polynomials import _discrete_antiderivative as module

    source = _polynomial(((1, (40, 0)),))
    cancelled = Event()

    def checkpoint(stage: str) -> None:
        request_checkpoint(stage)
        if stage == "during discrete antiderivative solve":
            cancelled.set()

    monkeypatch.setattr(module, "request_checkpoint", checkpoint)
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError),
    ):
        rational_discrete_antiderivative(source, "k")


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
        "variable",
        "antiderivative",
        "reconstructed_difference",
    }
