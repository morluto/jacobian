from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from sympy import Poly, apart, symbols

from jacobian.math import polynomials


def test_native_polynomial_api_uses_exact_sympy_values() -> None:
    x = symbols("x")
    left = Poly(x**2 - 1, x, domain="QQ")
    right = Poly(x - 1, x, domain="QQ")
    left_multiplier, right_multiplier, gcd = polynomials.gcdex(left, right)
    assert left * left_multiplier + right * right_multiplier == gcd == right
    assert polynomials.derivative(left) == Poly(2 * x, x, domain="QQ")
    assert polynomials.discriminant(left, x) == 4
    quotient, remainder, reconstruction = polynomials.divide(left, right)
    assert quotient == Poly(x + 1, x, domain="QQ") and remainder.is_zero
    assert reconstruction == left and polynomials.evaluate(left, 2) == 3
    coefficient, factors, reconstructed = polynomials.factorization(left)
    assert coefficient == 1 and reconstructed == left
    assert {factor.as_expr() for factor, _ in factors} == {x - 1, x + 1}
    assert polynomials.groebner_basis((left, right), (x,), "lex") == (right,)
    assert polynomials.integral(right) == Poly(x**2 / 2 - x, x, domain="QQ")
    assert polynomials.partial_fractions(1 / (x * (x + 1)), x) == apart(
        1 / (x * (x + 1)), x
    )
    coefficient, factors, reconstructed = polynomials.square_free_decomposition(left)
    assert coefficient == 1 and factors == ((left, 1),) and reconstructed == left
    assert polynomials.resultant(left, right, x) == 0


def test_native_resultant_preserves_source_orientation() -> None:
    x = symbols("x")
    linear = Poly(x + 2, x, domain="QQ")
    cubic = Poly(x**3 + 1, x, domain="QQ")
    assert polynomials.resultant(linear, cubic, x) == -7
    assert polynomials.resultant(cubic, linear, x) == 7


@pytest.mark.parametrize(
    "decompose",
    (polynomials.factorization, polynomials.square_free_decomposition),
)
def test_native_polynomial_decompositions_preserve_integer_leading_content(
    decompose: Callable[[Poly], tuple[Any, tuple[tuple[Poly, int], ...], Poly]],
) -> None:
    x = symbols("x")
    source = Poly((2 * x + 1) ** 2, x, domain="ZZ")
    coefficient, factors, reconstructed = decompose(source)
    assert coefficient == 4
    assert factors == ((Poly(2 * x + 1, x, domain="ZZ").monic(), 2),)
    assert reconstructed == source


def test_native_groebner_basis_rejects_non_rational_domains() -> None:
    x, y = symbols("x y")
    generators = (Poly(x + y, x, y, modulus=2), Poly(x - y, x, y, modulus=2))
    with pytest.raises(ValueError, match="QQ domain"):
        polynomials.groebner_basis(generators, (x, y), "lex")


def test_native_discriminant_preserves_the_polynomial_domain() -> None:
    x = symbols("x")
    assert polynomials.discriminant(Poly(x**2 + x + 1, x, modulus=2), x) == 1


def test_exact_public_api_symbols() -> None:
    expected = (
        "derivative",
        "discriminant",
        "divide",
        "evaluate",
        "factorization",
        "gcdex",
        "groebner_basis",
        "hermite_reduction",
        "integer_polynomial_compose",
        "integer_polynomial_content",
        "integer_polynomial_evaluate",
        "integer_polynomial_gcd",
        "integer_polynomial_primitive_part",
        "integer_polynomial_shift",
        "integral",
        "multiply",
        "partial_fractions",
        "polynomial_discriminant",
        "polynomial_factorization",
        "polynomial_gcd",
        "polynomial_groebner_basis",
        "polynomial_resultant",
        "polynomial_square_free_decomposition",
        "rational_partial_fraction_decomposition",
        "rational_polynomial_derivative",
        "rational_polynomial_division",
        "rational_polynomial_evaluate",
        "rational_polynomial_integral",
        "resultant",
        "square_free_decomposition",
    )
    assert tuple(polynomials.__all__) == expected
    assert len(polynomials.__all__) == len(set(polynomials.__all__))
    assert all(
        not name.startswith("_") and hasattr(polynomials, name) for name in expected
    )


def _univariate(variable: str, terms: dict[int, str]) -> Any:
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    return RationalPolynomial(
        variables=(variable,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(
                        num=value.split("/")[0],
                        den=value.split("/")[1] if "/" in value else "1",
                    ),
                    exponents=(degree,),
                )
                for degree, value in sorted(terms.items(), reverse=True)
            )
        ),
    )


def test_factor_producers_compute_once_and_round_trip_structurally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import operations as _operations
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialSquareFreeDecompositionResult,
    )

    source = _univariate("x", {4: "1", 2: "-2", 0: "1"})
    factor_calls = square_free_calls = 0
    original_factorization = _operations.factorization
    original_square_free = _operations.square_free_decomposition

    def count_factorization(poly: Any) -> Any:
        nonlocal factor_calls
        factor_calls += 1
        return original_factorization(poly)

    def count_square_free(poly: Any) -> Any:
        nonlocal square_free_calls
        square_free_calls += 1
        return original_square_free(poly)

    monkeypatch.setattr(_operations, "factorization", count_factorization)
    monkeypatch.setattr(_operations, "square_free_decomposition", count_square_free)
    factorization = _operations.polynomial_factorization(source)
    square_free = _operations.polynomial_square_free_decomposition(source)
    assert (factor_calls, square_free_calls) == (1, 1)
    assert factorization.reconstructed == square_free.reconstructed == source
    assert (
        PolynomialFactorizationResult.model_validate(factorization.model_dump())
        == factorization
    )
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(square_free.model_dump())
        == square_free
    )


def test_factor_results_keep_structural_ring_and_order_checks() -> None:
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialIrreducibleFactor,
    )
    from jacobian.math.polynomials.operations import polynomial_factorization

    result = polynomial_factorization(_univariate("x", {3: "1", 0: "-1"}))
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult(
            polynomial=result.polynomial,
            coefficient=result.coefficient,
            factors=tuple(reversed(result.factors)),
            reconstructed=result.reconstructed,
        )
    with pytest.raises(ValidationError, match="reconstructed polynomial"):
        PolynomialFactorizationResult(
            polynomial=result.polynomial,
            coefficient=result.coefficient,
            factors=(),
            reconstructed=_univariate("y", {3: "1", 0: "-1"}),
        )
    foreign = PolynomialIrreducibleFactor(
        factor=_univariate("y", {1: "1", 0: "-1"}), multiplicity=1
    )
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult(
            polynomial=result.polynomial,
            coefficient=result.coefficient,
            factors=(foreign,),
            reconstructed=result.reconstructed,
        )
