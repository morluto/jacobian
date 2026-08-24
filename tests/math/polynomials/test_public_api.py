from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
from sympy import Poly, apart, symbols

from jacobian.math import polynomials


def test_native_polynomial_api_uses_exact_sympy_values() -> None:
    x = symbols("x")
    left = Poly(x**2 - 1, x, domain="QQ")
    right = Poly(x - 1, x, domain="QQ")

    left_multiplier, right_multiplier, gcd = polynomials.gcdex(left, right)
    assert left * left_multiplier + right * right_multiplier == gcd
    assert gcd == right
    assert polynomials.derivative(left) == Poly(2 * x, x, domain="QQ")
    assert polynomials.discriminant(left, x) == 4
    quotient, remainder, reconstruction = polynomials.divide(left, right)
    assert quotient == Poly(x + 1, x, domain="QQ")
    assert remainder.is_zero
    assert reconstruction == left
    assert polynomials.evaluate(left, 2) == 3
    coefficient, factors, reconstructed = polynomials.factorization(left)
    assert coefficient == 1
    assert reconstructed == left
    assert {factor.as_expr() for factor, _multiplicity in factors} == {x - 1, x + 1}
    assert polynomials.groebner_basis((left, right), (x,), "lex") == (right,)
    assert polynomials.integral(right) == Poly(x**2 / 2 - x, x, domain="QQ")
    assert polynomials.partial_fractions(1 / (x * (x + 1)), x) == apart(
        1 / (x * (x + 1)), x
    )
    square_free_coefficient, square_free_factors, square_free_reconstruction = (
        polynomials.square_free_decomposition(left)
    )
    assert square_free_coefficient == 1
    assert square_free_factors == ((left, 1),)
    assert square_free_reconstruction == left
    assert polynomials.resultant(left, right, x) == 0


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
    generators = (
        Poly(x + y, x, y, modulus=2),
        Poly(x - y, x, y, modulus=2),
    )

    with pytest.raises(ValueError, match="QQ domain"):
        polynomials.groebner_basis(generators, (x, y), "lex")


def test_native_discriminant_preserves_the_polynomial_domain() -> None:
    x = symbols("x")
    polynomial = Poly(x**2 + x + 1, x, modulus=2)

    assert polynomials.discriminant(polynomial, x) == 1


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the polynomials public API."""
    expected = (
        "derivative",
        "discriminant",
        "divide",
        "evaluate",
        "factorization",
        "gcdex",
        "groebner_basis",
        "integral",
        "partial_fractions",
        "resultant",
        "square_free_decomposition",
    )
    assert tuple(polynomials.__all__) == expected
    assert len(polynomials.__all__) == len(set(polynomials.__all__))
    assert all(not name.startswith("_") for name in polynomials.__all__)
    assert all(hasattr(polynomials, name) for name in polynomials.__all__)


# ---------------------------------------------------------------------------
# Source-bound factorization replay (#2298)
# ---------------------------------------------------------------------------


def _univariate(variable: str, terms: dict[int, str]):
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    ordered = sorted(terms.items(), reverse=True)
    return RationalPolynomial(
        variables=(variable,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(
                        num=v.split("/")[0], den=(v.split("/")[1] if "/" in v else "1")
                    ),
                    exponents=(degree,),
                )
                for degree, v in ordered
            )
        ),
    )


def test_factorization_results_replay_against_source() -> None:
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    cases = {
        "constant": ("x", {0: "6"}),
        "irreducible": ("x", {2: "1", 1: "1", 0: "1"}),
        "repeated_pure_power": ("x", {4: "1", 3: "4", 2: "6", 1: "4", 0: "1"}),
        "rational_content": ("x", {3: "9/4", 2: "3/2", 0: "-5/4"}),
        "sign_normalization": ("x", {2: "-4", 0: "-1"}),
    }
    for _label, (variable, terms) in cases.items():
        request = _univariate(variable, terms)
        result = polynomial_factorization(PolynomialFactorRequest(polynomial=request))
        assert result.polynomial == request
        assert result.reconstructed == request
        product = Fraction(int(result.coefficient.num), int(result.coefficient.den))
        for record in result.factors:
            factor_value = _evaluate(record.factor, 7)
            product *= factor_value**record.multiplicity
        source_value = _evaluate(request, 7)
        reconstructed_value = _evaluate(result.reconstructed, 7)
        assert reconstructed_value == source_value
        assert product == source_value
        assert (
            PolynomialFactorizationResult.model_validate(result.model_dump()) == result
        )


def test_square_free_decomposition_replays_against_source() -> None:
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    request = _univariate("x", {4: "2", 3: "4", 2: "2"})
    result = polynomial_square_free_decomposition(
        PolynomialSquareFreeRequest(polynomial=request)
    )
    assert result.polynomial == request
    assert result.reconstructed == request
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def _evaluate(polynomial, point: int) -> Fraction:
    value = Fraction(0)
    for term in polynomial.polynomial.terms:
        value += (
            Fraction(int(term.coefficient.num), int(term.coefficient.den))
            * point ** term.exponents[0]
        )
    return value


def test_factorization_result_rejects_mutations() -> None:
    import copy as _copy

    from pydantic import ValidationError

    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    request = _univariate("x", {2: "1", 0: "-1"})
    result = polynomial_factorization(PolynomialFactorRequest(polynomial=request))
    dumped = result.model_dump()

    coefficient_mutation = _copy.deepcopy(dumped)
    coefficient_mutation["coefficient"] = {"num": "2", "den": "1"}
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult.model_validate(coefficient_mutation)

    multiplicity_mutation = _copy.deepcopy(dumped)
    multiplicity_mutation["factors"][0]["multiplicity"] += 1
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult.model_validate(multiplicity_mutation)

    factor_term_mutation = _copy.deepcopy(dumped)
    factor_term_mutation["factors"][0]["factor"]["polynomial"]["terms"][0][
        "exponents"
    ] = [3]
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult.model_validate(factor_term_mutation)

    reconstructed_mutation = _copy.deepcopy(dumped)
    reconstructed_mutation["reconstructed"]["polynomial"]["terms"] = []
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult.model_validate(reconstructed_mutation)

    source_mutation = _copy.deepcopy(dumped)
    source_mutation["polynomial"]["polynomial"]["terms"] = []
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult.model_validate(source_mutation)


def test_equivalent_factor_orders_normalize_canonically() -> None:
    """A hand-built result with backend-incidental ordering still replays."""

    from pydantic import ValidationError

    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    request = _univariate("x", {3: "1", 0: "-1"})
    produced = polynomial_factorization(PolynomialFactorRequest(polynomial=request))
    if len(produced.factors) > 1:
        with pytest.raises(ValidationError, match="ordered by"):
            PolynomialFactorizationResult(
                polynomial=produced.polynomial,
                coefficient=produced.coefficient,
                factors=tuple(reversed(produced.factors)),
                reconstructed=produced.reconstructed,
            )
    canonical = PolynomialFactorizationResult(
        polynomial=produced.polynomial,
        coefficient=produced.coefficient,
        factors=tuple(sorted(produced.factors, key=_canonical_key)),
        reconstructed=produced.reconstructed,
    )
    assert canonical == produced


def _canonical_key(record):
    return (
        record.multiplicity,
        max((sum(t.exponents) for t in record.factor.polynomial.terms), default=0),
        tuple(
            (t.exponents, t.coefficient.num, t.coefficient.den)
            for t in record.factor.polynomial.terms
        ),
    )
