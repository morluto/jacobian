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


def _sparse_polynomial(variables: tuple[str, ...], terms: dict[tuple[int, ...], str]):
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=terms[exponents], den="1"),
                    exponents=exponents,
                )
                for exponents in sorted(terms, reverse=True)
            )
        ),
    )


def _source_bound_result_cases() -> tuple[tuple[Callable[[Any], Any], type, type], ...]:
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_factorization,
        polynomial_square_free_decomposition,
    )

    return (
        (
            polynomial_factorization,
            PolynomialFactorRequest,
            PolynomialFactorizationResult,
        ),
        (
            polynomial_square_free_decomposition,
            PolynomialSquareFreeRequest,
            PolynomialSquareFreeDecompositionResult,
        ),
    )


def _forged_monster_payload(dumped: dict[str, Any]) -> dict[str, Any]:
    import copy as _copy

    monster = _sparse_polynomial(
        ("x", "y"),
        {(first, second): "1" for first in range(16) for second in range(16)},
    )
    source = _sparse_polynomial(("x", "y"), {(2, 0): "1", (0, 2): "1"})
    forged = _copy.deepcopy(dumped)
    forged["polynomial"] = source.model_dump()
    forged["reconstructed"] = source.model_dump()
    forged["factors"] = [{"factor": monster.model_dump(), "multiplicity": 64}]
    return forged


def test_source_bound_results_reject_combinatorial_replay_claims() -> None:
    """Forged factor records are rejected without expanding their product.

    The claimed factor is a dense-box multivariate sum within the shared wire
    limits whose multiplicity-64 power cannot be expanded; validation must
    reject it at the first inexact division instead.
    """

    from pydantic import ValidationError

    for operation, request_model, result_model in _source_bound_result_cases():
        request = request_model(polynomial=_univariate("x", {2: "1", 0: "-1"}))
        produced = operation(request)
        forged = _forged_monster_payload(produced.model_dump())
        with pytest.raises(ValidationError, match="must reconstruct"):
            result_model.model_validate(forged)


def test_zero_content_results_retain_no_factors() -> None:
    """A zero coefficient cannot authenticate an arbitrary factor list."""

    from pydantic import ValidationError

    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    zero = _univariate("x", {})
    produced = polynomial_factorization(PolynomialFactorRequest(polynomial=zero))
    assert produced.coefficient.as_fraction() == 0
    assert PolynomialFactorizationResult.model_validate(produced.model_dump())

    forged = produced.model_dump()
    forged["factors"] = [
        {
            "factor": _univariate("x", {1: "1", 0: "-1"}).model_dump(),
            "multiplicity": 1,
        }
    ]
    with pytest.raises(ValidationError, match="zero content"):
        PolynomialFactorizationResult.model_validate(forged)


def test_factorization_replays_at_the_request_degree_cap() -> None:
    """A pure power at the degree-127 request cap still replays exactly."""

    from sympy import Poly, symbols

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    x = symbols("x")
    expanded = Poly((x + 1) ** 127, x, domain="QQ")
    coefficients = [str(int(value)) for value in expanded.all_coeffs()]
    request = PolynomialFactorRequest(
        polynomial=_univariate(
            "x",
            {
                degree: coefficients[index]
                for index, degree in enumerate(range(127, -1, -1))
            },
        )
    )
    result = polynomial_factorization(request)
    records = [
        (rational_polynomial_to_sympy(record.factor).as_expr(), record.multiplicity)
        for record in result.factors
    ]
    assert records == [(x + 1, 127)]
    assert PolynomialFactorizationResult.model_validate(result.model_dump()) == result


def test_square_free_replays_at_the_multiplicity_cap() -> None:
    """A pure power at the multiplicity-64 square-free cap still replays."""

    from sympy import Poly, symbols

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    x = symbols("x")
    expanded = Poly((x + 1) ** 64, x, domain="QQ")
    coefficients = [str(int(value)) for value in expanded.all_coeffs()]
    request = PolynomialSquareFreeRequest(
        polynomial=_univariate(
            "x",
            {
                degree: coefficients[index]
                for index, degree in enumerate(range(64, -1, -1))
            },
        )
    )
    result = polynomial_square_free_decomposition(request)
    records = [
        (rational_polynomial_to_sympy(record.factor).as_expr(), record.multiplicity)
        for record in result.factors
    ]
    assert records == [(x + 1, 64)]
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_result_models_parse_coefficients_above_int_str_limit() -> None:
    """Coefficients past CPython's 4,300-digit int-str limit replay exactly.

    ``CanonicalRational`` admits up to 32,768 digits through the chunked
    canonical parser; the replay must convert it via ``as_fraction`` rather
    than a direct ``int(...)`` cast.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialSquareFreeDecompositionResult,
    )

    digits = 5_000
    assert digits > 4_300
    source = _univariate("x", {0: "9" * digits})
    models = (
        PolynomialFactorizationResult,
        PolynomialSquareFreeDecompositionResult,
    )
    for model in models:
        result = model(
            polynomial=source,
            coefficient=CanonicalRational(num="9" * digits, den="1"),
            factors=(),
            reconstructed=source,
        )
        assert model.model_validate(result.model_dump()) == result
