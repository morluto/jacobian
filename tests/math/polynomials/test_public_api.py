from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
from sympy import Poly, apart, symbols
from tests.math.polynomials._support import polynomial_validation_error

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


def test_native_resultant_preserves_source_orientation() -> None:
    """Unequal odd degrees retain the Sylvester determinant sign."""

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
        "hermite_reduction",
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


def test_factorization_results_reject_multivariate_sources() -> None:
    """The univariate request domain also binds an authored or replayed result.

    A QQ[x, y] payload whose exact-division replay succeeds (source ``x*y``
    with factors ``x`` and ``y``) must still be rejected because
    ``polynomial.factor.compute`` factorizes univariate polynomials only.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialIrreducibleFactor,
    )

    variables = ("x", "y")
    source = _sparse_polynomial(variables, {(1, 1): "1"})
    factors = (
        PolynomialIrreducibleFactor(
            factor=_sparse_polynomial(variables, {(1, 0): "1"}), multiplicity=1
        ),
        PolynomialIrreducibleFactor(
            factor=_sparse_polynomial(variables, {(0, 1): "1"}), multiplicity=1
        ),
    )
    with polynomial_validation_error():
        PolynomialFactorizationResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=factors,
            reconstructed=source,
        )


def test_equivalent_factor_orders_normalize_canonically() -> None:
    """A hand-built result with backend-incidental ordering still replays."""

    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialFactorRequest,
    )
    from jacobian.math.polynomials._operations import polynomial_factorization

    request = _univariate("x", {3: "1", 0: "-1"})
    produced = polynomial_factorization(PolynomialFactorRequest(polynomial=request))
    if len(produced.factors) > 1:
        with polynomial_validation_error():
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
    limits whose multiplicity-64 power chain exhausts the cumulative
    reconstruction-work budget; bounded reconstruction counts each step's
    exact work and stops typedly before any oversized expansion, with every
    executed step's materialized support inside the per-step ceiling.
    """

    for operation, request_model, result_model in _source_bound_result_cases():
        request = request_model(polynomial=_univariate("x", {2: "1", 0: "-1"}))
        produced = operation(request)
        forged = _forged_monster_payload(produced.model_dump())
        # The factorization contract is univariate, so a forged QQ[x, y]
        # source is rejected by the semantic-domain check before any replay
        # work; square-free decomposition admits several variables, and the
        # monster's multiplicity-64 power chain keeps every per-step degree
        # box inside the replay ceiling (its dense grids collapse into
        # predictable boxes) while the cumulative monomial-multiplication
        # count crosses the reconstruction budget, so the work guard rejects
        # the claim before any further multiplication could run.
        with polynomial_validation_error():
            result_model.model_validate(forged)


def test_zero_content_results_retain_no_factors() -> None:
    """A zero coefficient cannot authenticate an arbitrary factor list."""

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
    with polynomial_validation_error():
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


def test_result_models_bind_sources_to_request_coefficient_budgets() -> None:
    """Retained sources answer to the same coefficient budget as requests.

    Both originating request models admit only the default 256-digit
    polynomial coefficients, so an authored or deserialized result whose
    retained source carries more digits must be rejected at admission
    before any replay work instead of widening the operation's established
    work envelope; sources within the budget round-trip exactly.
    """

    from pydantic import ValidationError

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialSquareFreeDecompositionResult,
    )

    models = (
        PolynomialFactorizationResult,
        PolynomialSquareFreeDecompositionResult,
    )
    for model in models:
        source = _univariate("x", {0: "9" * 257})
        with pytest.raises(ValidationError):
            model(
                polynomial=source,
                coefficient=CanonicalRational(num="9" * 257, den="1"),
                factors=(),
                reconstructed=source,
            )
        source = _univariate("x", {0: "9" * 256})
        result = model(
            polynomial=source,
            coefficient=CanonicalRational(num="9" * 256, den="1"),
            factors=(),
            reconstructed=source,
        )
        assert model.model_validate(result.model_dump()) == result


# ---------------------------------------------------------------------------
# Bounded source-bound replay (#2298)
# ---------------------------------------------------------------------------


def _multivariate_binomial_box(variables: tuple[str, ...], exponent: int) -> Any:
    """``prod(x_i ** exponent - 1)`` or ``prod(x_i - 1)`` as a wire value."""

    import itertools

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    def build(exponent_for_selected: int) -> RationalPolynomial:
        terms = {}
        for mask in itertools.product((False, True), repeat=len(variables)):
            exponents = tuple(
                exponent_for_selected if selected else 0 for selected in mask
            )
            terms[exponents] = "-1" if sum(mask) % 2 else "1"
        return RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=c, den="1"),
                        exponents=e,
                    )
                    for e, c in sorted(terms.items(), reverse=True)
                )
            ),
        )

    return build(exponent_for_selected=exponent), build(exponent_for_selected=1)


def test_square_free_replay_rejects_dense_multivariate_claims_without_dividing() -> (
    None
):
    """The eight-variable binomial product payload is rejected before expansion.

    The 256-term source ``prod(v_i^64 - 1)`` and the 256-term claimed factor
    ``prod(v_i - 1)`` both pass the shared wire budgets, but their exact
    quotient ``prod(1 + v_i + ... + v_i^63)`` would carry ``64^8`` monomials.
    Multivariate validation therefore never divides: it recomputes the
    canonical monic decomposition of the retained source and rejects the
    mismatched records typedly without materializing any quotient.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    variables = tuple(f"v{index}" for index in range(8))
    source, factor = _multivariate_binomial_box(variables, 64)
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialSquareFreeFactor(factor=factor, multiplicity=1),),
            reconstructed=source,
        )


def test_multivariate_square_free_results_still_replay_against_source() -> None:
    """Multivariate decompositions keep validating under bounded replay.

    The canonical recomparison lane must admit exactly what the producer
    emits: a pure power at the former degree-box boundary, a smaller
    multiplicity-bearing bivariate part, and a content-carrying
    decomposition round-trip through the operation and the validator.
    """

    from math import comb

    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    requests = (
        _sparse_polynomial(
            ("x", "y"),
            {(64 - k, k): str(comb(64, k)) for k in range(65)},
        ),
        _sparse_polynomial(("x", "y"), {(2, 2): "1", (1, 1): "-2", (0, 0): "1"}),
        _sparse_polynomial(
            ("x", "y"),
            {(2, 2): "3", (2, 0): "-3", (0, 2): "-3", (0, 0): "3"},
        ),
    )
    expected_records = ([(2, 64)], [(2, 2)], [(4, 1)])
    for request, expected in zip(requests, expected_records, strict=True):
        result = polynomial_square_free_decomposition(
            PolynomialSquareFreeRequest(polynomial=request)
        )
        records = [
            (len(record.factor.polynomial.terms), record.multiplicity)
            for record in result.factors
        ]
        assert records == expected
        assert result.reconstructed == result.polynomial
        assert (
            PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
            == result
        )


def test_square_free_replay_admits_sparse_cofactors_above_the_division_box() -> None:
    """A legitimate output whose degree box exceeds the envelope validates.

    For ``(x_1^16 + x_3^16) * (x_2 + ... + x_8)^3`` the first replay
    division's degree box is ``4^7 = 16384`` terms while the true cofactor
    is the 84-term cube, so an envelope-rejected division replay crashed
    the operation on this admitted 168-term request.  The recomparison
    lane verifies such outputs against the recomputed canonical parts
    without dividing, and a forged claim over the same source still fails.
    """

    from collections import defaultdict
    from itertools import repeat

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    def multiply(left, right):
        product: dict[tuple[int, ...], int] = defaultdict(int)
        for left_exponents, left_coefficient in left.items():
            for right_exponents, right_coefficient in right.items():
                product[
                    tuple(
                        a + b
                        for a, b in zip(left_exponents, right_exponents, strict=True)
                    )
                ] += left_coefficient * right_coefficient
        return dict(product)

    size = 8
    linear = {
        tuple(1 if index == j else 0 for index in range(size)): 1
        for j in range(1, size)
    }
    cube = {tuple(repeat(0, size)): 1}
    for _ in range(3):
        cube = multiply(cube, linear)
    binomials = {
        tuple(16 if index == peak else 0 for index in range(size)): 1 for peak in (0, 2)
    }
    source_terms = multiply(binomials, cube)
    assert len(source_terms) == 168
    source = _sparse_polynomial(
        tuple(f"x{index}" for index in range(1, size + 1)),
        {exponents: str(value) for exponents, value in source_terms.items()},
    )
    result = PolynomialSquareFreeDecompositionResult(
        polynomial=source,
        coefficient=CanonicalRational(num="1", den="1"),
        factors=(
            PolynomialSquareFreeFactor(
                factor=_sparse_polynomial(
                    source.variables,
                    {exponents: str(value) for exponents, value in binomials.items()},
                ),
                multiplicity=1,
            ),
            PolynomialSquareFreeFactor(
                factor=_sparse_polynomial(
                    source.variables,
                    {exponents: str(value) for exponents, value in linear.items()},
                ),
                multiplicity=3,
            ),
        ),
        reconstructed=source,
    )
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )

    forged_factor = _sparse_polynomial(source.variables, {(1,) * size: "1"})
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialSquareFreeFactor(factor=forged_factor, multiplicity=1),),
            reconstructed=source,
        )


def test_square_free_results_require_canonical_univariate_parts() -> None:
    """A product-correct but overlapping univariate grouping is not the result.

    ``(x - 1) * ((x - 1)(x + 1))**2`` equals the retained source
    ``(x - 1)**3 * (x + 1)**2``, so every univariate exact-division step
    succeeds while the claimed parts overlap; a scalar shuffle across the
    records also reconstructs exactly yet breaks ``MONIC_FACTORS``.  Only
    the recomputed canonical monic square-free parts authenticate either
    claim.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    source = _univariate("x", {5: "1", 4: "-1", 3: "-2", 2: "2", 1: "1", 0: "-1"})
    linear = _univariate("x", {1: "1", 0: "-1"})
    mixed = _univariate("x", {2: "1", 0: "-1"})
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=linear, multiplicity=1),
                PolynomialSquareFreeFactor(factor=mixed, multiplicity=2),
            ),
            reconstructed=source,
        )
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="2"),
            factors=(
                PolynomialSquareFreeFactor(
                    factor=_univariate("x", {1: "1/2", 0: "1/2"}), multiplicity=2
                ),
                PolynomialSquareFreeFactor(
                    factor=_univariate("x", {1: "2", 0: "-2"}), multiplicity=3
                ),
            ),
            reconstructed=source,
        )


def test_square_free_operation_round_trips_distinct_univariate_multiplicities() -> None:
    """The producer's records stay canonical under univariate revalidation.

    ``(x - 1)**3 * (x + 1)**2`` decomposes into the distinct-multiplicity
    parts ``(x + 1, 2)`` and ``(x - 1, 3)``, so the canonical-recomparison
    lane must admit exactly that emitted decomposition and its serialized
    round trip.
    """

    from sympy import symbols

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    request = _univariate("x", {5: "1", 4: "-1", 3: "-2", 2: "2", 1: "1", 0: "-1"})
    result = polynomial_square_free_decomposition(
        PolynomialSquareFreeRequest(polynomial=request)
    )
    x = symbols("x")
    records = [
        (rational_polynomial_to_sympy(record.factor).as_expr(), record.multiplicity)
        for record in result.factors
    ]
    assert records == [(x + 1, 2), (x - 1, 3)]
    assert result.reconstructed == request
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_square_free_results_require_canonical_square_free_parts() -> None:
    """A product-correct but non-coprime grouping is not the decomposition.

    ``(x - 1) * ((x - 1)(y - 1))**2`` equals the retained source
    ``(x - 1)**3 * (y - 1)**2``, so every exact-division step would
    succeed; only the unique canonical monic square-free parts
    authenticate the claimed records.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    source = _sparse_polynomial(
        ("x", "y"),
        {
            (3, 2): "1",
            (3, 1): "-2",
            (3, 0): "1",
            (2, 2): "-3",
            (2, 1): "6",
            (2, 0): "-3",
            (1, 2): "3",
            (1, 1): "-6",
            (1, 0): "3",
            (0, 2): "-1",
            (0, 1): "2",
            (0, 0): "-1",
        },
    )
    linear = _sparse_polynomial(("x", "y"), {(1, 0): "1", (0, 0): "-1"})
    mixed = _sparse_polynomial(
        ("x", "y"), {(1, 1): "1", (1, 0): "-1", (0, 1): "-1", (0, 0): "1"}
    )
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=linear, multiplicity=1),
                PolynomialSquareFreeFactor(factor=mixed, multiplicity=2),
            ),
            reconstructed=source,
        )


def test_square_free_replay_admits_sparse_multivariate_pure_powers() -> None:
    """``x^64 * y^64 * z^64`` decomposes and validates as ``(x*y*z)^64``.

    A degree-box replay bound would project a ``64^3``-term first quotient
    for the one-term source and reject this admitted request; the
    recomparison lane returns and revalidates the producer's records.
    """

    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    result = polynomial_square_free_decomposition(
        PolynomialSquareFreeRequest(
            polynomial=_sparse_polynomial(("x", "y", "z"), {(64, 64, 64): "1"})
        )
    )
    assert [
        (len(record.factor.polynomial.terms), record.multiplicity)
        for record in result.factors
    ] == [(1, 64)]
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_square_free_replay_returns_typed_records_for_trivariate_residual() -> None:
    """``(x*y*z - 1)**20 * (x + y + z)`` returns a typed decomposition.

    This admitted 63-term trivariate request was the residual host-exception
    report against the division-based replay lanes; canonical recomparison
    must return the producer's records and revalidate their serialized form
    instead of raising on an accepted request.
    """

    from math import comb

    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    terms: dict[tuple[int, int, int], str] = {}
    for power in range(21):
        coefficient = comb(20, power) * (-1) ** (20 - power)
        for shifted in range(3):
            exponents = [power, power, power]
            exponents[shifted] += 1
            terms[tuple(exponents)] = str(coefficient)
    assert len(terms) == 63
    request = _sparse_polynomial(("x", "y", "z"), terms)
    result = polynomial_square_free_decomposition(
        PolynomialSquareFreeRequest(polynomial=request)
    )
    assert [
        (len(record.factor.polynomial.terms), record.multiplicity)
        for record in result.factors
    ] == [(3, 1), (2, 20)]
    assert result.reconstructed == request
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_square_free_replay_rejects_inexact_multivariate_divisors() -> None:
    """A forged divisor with a tiny degree box is rejected without dividing.

    The claimed factor ``v_0 - sum(v_i^64)`` over eight variables leaves
    per-variable degree room ``64`` and ``1`` against the monomial source
    ``prod(v_i^64)``, so a division replay would pass any box bound while
    its inexact lexicographic long division builds huge partial sums.  The
    recomparison lane rejects the mismatched records typedly instead.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    variables = tuple(f"v{index}" for index in range(8))
    source = _sparse_polynomial(variables, {(64,) * 8: "1"})
    terms = {(1, *(0,) * 7): "1"}
    for index in range(1, 8):
        terms[tuple(64 if position == index else 0 for position in range(8))] = "-1"
    factor = _sparse_polynomial(variables, terms)
    assert len(factor.polynomial.terms) == 8
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialSquareFreeFactor(factor=factor, multiplicity=1),),
            reconstructed=source,
        )


def test_factorization_replay_requires_canonical_irreducible_records() -> None:
    """A reducible record cannot authenticate an irreducible factorization.

    ``x**2 - 1`` reconstructs itself exactly when claimed as the single
    "irreducible" factor of the equal source, so a product-only replay
    would accept it; re-deriving the unique content-and-monic-irreducibles
    factorization rejects the non-canonical multiset typedly.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialIrreducibleFactor,
    )

    source = _univariate("x", {2: "1", 0: "-1"})
    with polynomial_validation_error():
        PolynomialFactorizationResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialIrreducibleFactor(factor=source, multiplicity=1),),
            reconstructed=source,
        )


def _difference_product(variables: tuple[str, ...], exponent: int) -> Any:
    """``prod(v_i ** exponent - 1)`` as a wire value with descending terms."""

    import itertools

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    terms = {}
    for mask in itertools.product((False, True), repeat=len(variables)):
        exponents = tuple(exponent if selected else 0 for selected in mask)
        terms[exponents] = "-1" if sum(mask) % 2 else "1"
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=c, den="1"), exponents=e
                )
                for e, c in sorted(terms.items(), reverse=True)
            )
        ),
    )


def test_square_free_replay_never_recomputes_unrepresentable_decompositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation rejects claims without materializing the true decomposition.

    The five-variable source ``prod((x_i^63 - 1)(x_i - 1))`` holds exactly
    1,024 terms inside the request envelope, yet its canonical
    multiplicity-one part ``prod(1 + x_i + ... + x_i^62)`` carries ``63^5``
    terms — far beyond any representable bound.  A forged or deserialized
    result over this source must therefore be rejected by bounded checks on
    the claimed records alone; no backend decomposition may run at all.
    """

    from sympy import Poly, symbols

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
    )

    variables = tuple(f"x{index}" for index in range(1, 6))
    generators = symbols(",".join(variables))
    source_expr = Poly(1, *generators, domain="QQ")
    for symbol in generators:
        source_expr *= (symbol**63 - 1) * (symbol - 1)
    assert len(source_expr.terms()) == 1024
    source = _sparse_polynomial(
        variables,
        {monom: str(int(coeff)) for monom, coeff in source_expr.terms()},
    )

    def fail_decomposition(self):
        raise AssertionError("replay must not recompute any decomposition")

    monkeypatch.setattr(Poly, "sqf_list", fail_decomposition)
    monkeypatch.setattr(Poly, "factor_list", fail_decomposition)
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(),
            reconstructed=source,
        )


def test_square_free_operation_admits_transiently_dense_prefixes() -> None:
    """An admitted request stays admitted when a replay prefix densifies.

    For the accepted source ``A(x) * B(y, z)^2 * (x - 1)^3`` the canonical
    records hold only 62, 48, and 2 terms and the 930-term source fits its
    request envelope, yet multiplicity-ordered replay forms the transient
    prefix ``A * B^2`` with 9,610 terms before ``(x - 1)^3`` cancels it back
    into the envelope.  Replay prefixes are bounded per step by a dedicated
    replay ceiling rather than by the serialization envelope, so the
    operation must return its typed result instead of failing validation.
    """

    from sympy import Poly, symbols

    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeRequest,
    )
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    x, y, z = symbols("x,y,z")
    geometric_x = Poly(sum(x**i for i in range(62)), x, y, z, domain="QQ")
    geometric_yz = Poly(
        sum(y**j for j in range(16)) * sum(z**k for k in range(3)),
        x,
        y,
        z,
        domain="QQ",
    )
    source = Poly(
        (geometric_x * geometric_yz**2 * (x - 1) ** 3).as_expr(),
        x,
        y,
        z,
        domain="QQ",
    )
    assert len(source.terms()) == 930
    request = PolynomialSquareFreeRequest(
        polynomial=_sparse_polynomial(
            ("x", "y", "z"),
            {monom: str(int(coeff)) for monom, coeff in source.terms()},
        )
    )
    result = polynomial_square_free_decomposition(request)
    records = sorted(
        (len(rational_polynomial_to_sympy(r.factor).terms()), r.multiplicity)
        for r in result.factors
    )
    assert records == [(2, 3), (48, 2), (62, 1)]
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_square_free_operation_admits_telescoping_geometric_decomposition() -> None:
    """The reported admitted source returns its typed decomposition.

    ``S_57(x) * (S_31(y) S_31(z))^2 * (x+1)^3 * ((x-1)(y-1)(z-1))^4`` with
    ``S_n(t) = 1 + t + ... + t^(n-1)`` expands to 648 terms with per-variable
    degrees (63, 64, 64), inside the request envelope, and all four claimed
    canonical records fit their representation limits.  Its multiplicity-2
    record holds 961 terms whose squared support collapses to the full
    ``[0, 60]^2`` grid of 3,721 terms — a collision structure the former
    pairwise-product estimate (923,521 for that step alone) rejected as an
    intermediate overflow even though the authentic replay peaks at
    270,400 materialized terms and finishes inside the work budget.  The
    operation must return its typed, source-bound result.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials._models import PolynomialSquareFreeRequest
    from jacobian.math.polynomials._operations import (
        polynomial_square_free_decomposition,
    )

    def convolve_ints(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
        product: dict[int, int] = {}
        for left_degree, left_coeff in left.items():
            for right_degree, right_coeff in right.items():
                product[left_degree + right_degree] = (
                    product.get(left_degree + right_degree, 0)
                    + left_coeff * right_coeff
                )
        return {degree: c for degree, c in product.items() if c}

    def convolve(
        left: dict[tuple[int, ...], int], right: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, ...], int]:
        product: dict[tuple[int, ...], int] = {}
        for left_exps, left_coeff in left.items():
            for right_exps, right_coeff in right.items():
                key = tuple(a + b for a, b in zip(left_exps, right_exps, strict=True))
                product[key] = product.get(key, 0) + left_coeff * right_coeff
        return {exps: c for exps, c in product.items() if c}

    def on_axis(axis: int, part: dict[int, int]) -> dict[tuple[int, ...], int]:
        lifted: dict[tuple[int, ...], int] = {}
        for degree, coefficient in part.items():
            exps = [0, 0, 0]
            exps[axis] = degree
            lifted[tuple(exps)] = coefficient
        return lifted

    # Per-variable telescoped parts of the source:
    #   x: S_57(x)(x+1)^3(x-1)^4 = (x^57 - 1)(x^2 - 1)^3          -> 8 terms
    #   y: S_31(y)^2 (y-1)^4     = (y^62 - 2 y^31 + 1)(y^2-2y+1)  -> 9 terms
    #   z: same as y                                              -> 9 terms
    part_x = convolve_ints({57: 1, 0: -1}, {6: 1, 4: -3, 2: 3, 0: -1})
    part_y = convolve_ints({62: 1, 31: -2, 0: 1}, {2: 1, 1: -2, 0: 1})
    source_terms = convolve(
        convolve(on_axis(0, part_x), on_axis(1, part_y)), on_axis(2, part_y)
    )
    assert len(source_terms) == 648
    assert [max(exps[i] for exps in source_terms) for i in range(3)] == [63, 64, 64]
    request = PolynomialSquareFreeRequest(
        polynomial=_sparse_polynomial(
            ("x", "y", "z"),
            {exps: str(coefficient) for exps, coefficient in source_terms.items()},
        )
    )
    result = polynomial_square_free_decomposition(request)
    assert result.polynomial == request.polynomial
    assert result.reconstructed == request.polynomial
    assert result.coefficient == CanonicalRational(num="1", den="1")
    records = sorted(
        (
            (len(rational_polynomial_to_sympy(r.factor).terms()), r.multiplicity)
            for r in result.factors
        ),
        key=lambda record: record[1],
    )
    assert records == [(57, 1), (961, 2), (2, 3), (8, 4)]


def test_square_free_replay_rejects_forged_disjoint_grid_claims() -> None:
    """A forged disjoint-grid claim still rejects before materializing.

    Two schema-valid 1,296-term grid factors live on disjoint variable
    groups of one 8-variable ring at multiplicities 1 and 2.  Their merge's
    pairwise bound (about 19 million) and degree box (11^8) both exceed the
    per-step support ceiling, so the structure-derived preflight refuses the
    claim typedly before any backend multiplication can expand it.
    """

    import itertools

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    variables = tuple(f"x{index}" for index in range(8))
    grid = list(itertools.product(range(6), repeat=4))
    assert len(grid) == 1_296
    left = _sparse_polynomial(
        variables, {exponents + (0,) * 4: "1" for exponents in grid}
    )
    right = _sparse_polynomial(
        variables, {(0,) * 4 + exponents: "1" for exponents in grid}
    )
    constant = _sparse_polynomial(variables, {(0,) * 8: "1"})
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=constant,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=left, multiplicity=1),
                PolynomialSquareFreeFactor(factor=right, multiplicity=2),
            ),
            reconstructed=constant,
        )


def test_square_free_replay_support_prediction_keeps_colliding_claims_bounded() -> None:
    """Trusting the degree box admits only provably bounded products.

    A 729-term monic grid factor and a distinct 512-term grid factor live in
    one 3-variable ring at multiplicities 1 and 2.  The replay merge's
    pairwise bound (about 3.6 million) exceeds the per-step estimate that
    rejected authentic collisions before this fix, while the degree boxes
    prove every executed step's support stays inside the ceiling — so the
    steps run safely, and the forged constant-source claim is then rejected
    by the exact reconstruction equality instead.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    low_grid = {(a, b, c): "1" for a in range(9) for b in range(9) for c in range(9)}
    high_grid = {
        (a, b, c): "1" for a in range(9, 17) for b in range(9, 17) for c in range(9, 17)
    }
    assert len(low_grid) == 729 and len(high_grid) == 512
    first = _sparse_polynomial(("x", "y", "z"), low_grid)
    second = _sparse_polynomial(("x", "y", "z"), high_grid)
    constant = _sparse_polynomial(("x", "y", "z"), {(0, 0, 0): "1"})
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=constant,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=first, multiplicity=1),
                PolynomialSquareFreeFactor(factor=second, multiplicity=2),
            ),
            reconstructed=constant,
        )


def test_replay_preflights_support_before_multiplying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forged disjoint-support claims reject before any large materialization.

    Two schema-valid square-free factors each hold all 4,095 monomials of
    ``[0, 7]^4`` minus one on disjoint variable groups, at multiplicities 1
    and 2.  Their Cartesian merge has a 16,777,216-term degree box and a
    16,769,025-term pairwise bound, so both structure-derived support
    predictions exceed the per-step ceiling and the replay must refuse the
    merge before it materializes the product.  Squaring ``right`` is a
    different story its structure proves safe: the degree box of that step
    collapses to ``8^4 = 4,096`` terms, so it runs — and every operand of
    every multiplication that executes stays inside the ceiling.
    """

    import itertools

    from sympy import Poly

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials import _models
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    variables = tuple(f"x{index}" for index in range(8))
    box = [
        exponents
        for exponents in itertools.product(range(8), repeat=4)
        if exponents != (0, 0, 0, 0)
    ]
    assert len(box) == 4_095
    left = _sparse_polynomial(
        variables, {exponents + (0,) * 4: "1" for exponents in box}
    )
    right = _sparse_polynomial(
        variables, {(0,) * 4 + exponents: "1" for exponents in box}
    )
    constant = _sparse_polynomial(variables, {(0,) * 8: "1"})

    original_mul = Poly.__mul__
    observed_operands: list[int] = []

    def spy_mul(self, other):
        observed_operands.append(len(self.terms()))
        observed_operands.append(len(other.terms()))
        return original_mul(self, other)

    monkeypatch.setattr(Poly, "__mul__", spy_mul)
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=constant,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=left, multiplicity=1),
                PolynomialSquareFreeFactor(factor=right, multiplicity=2),
            ),
            reconstructed=constant,
        )
    # The left merge ran (trivially), then the right square ran under its
    # collapsed degree-box prediction, and the Cartesian merge was refused
    # before any backend multiplication: no operand ever exceeded the
    # materialization ceiling.
    assert observed_operands[-2:] == [4_095, 4_095]
    assert max(observed_operands) <= _models._MAX_REPLAY_INTERMEDIATE_TERMS


def test_square_free_operation_admits_envelope_scale_parts() -> None:
    """A 16-term admitted request yields its 3,969-term square-free part.

    For ``((x^63 - 1)(x - 1))((y^63 - 1)(y - 1))`` the multiplicity-one
    part ``((x^63 - 1)/(x - 1))((y^63 - 1)/(y - 1))`` holds 3,969 terms —
    inside the shared representation envelope and emitted by the backend —
    so an accepted request form of this decomposition must validate and
    round-trip instead of being rejected for its part size.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    geometric = _sparse_polynomial(
        ("x", "y"),
        {(a, b): "1" for a in range(63) for b in range(63)},
    )
    assert len(geometric.polynomial.terms) == 3_969
    # (x^64 - x^63 - x + 1)(y^64 - y^63 - y + 1), 16 terms, exponents <= 64.
    one_dimension = {0: 1, 1: -1, 63: -1, 64: 1}
    accumulated: dict[tuple[int, int], int] = {(0, 0): 1}
    for axis in (0, 1):
        shifted: dict[tuple[int, int], int] = {}
        for exps, coeff in accumulated.items():
            for degree, factor_coefficient in one_dimension.items():
                target = list(exps)
                target[axis] += degree
                key = tuple(target)
                shifted[key] = shifted.get(key, 0) + coeff * factor_coefficient
        accumulated = {exps: c for exps, c in shifted.items() if c}
    assert len(accumulated) == 16
    source = _sparse_polynomial(
        ("x", "y"),
        {exps: str(c) for exps, c in sorted(accumulated.items(), reverse=True)},
    )
    result = PolynomialSquareFreeDecompositionResult(
        polynomial=source,
        coefficient=CanonicalRational(num="1", den="1"),
        factors=(
            PolynomialSquareFreeFactor(factor=geometric, multiplicity=1),
            PolynomialSquareFreeFactor(
                factor=_sparse_polynomial(
                    ("x", "y"),
                    {(1, 1): "1", (1, 0): "-1", (0, 1): "-1", (0, 0): "1"},
                ),
                multiplicity=2,
            ),
        ),
        reconstructed=source,
    )
    assert (
        PolynomialSquareFreeDecompositionResult.model_validate(result.model_dump())
        == result
    )


def test_factorization_replay_rejects_split_duplicate_records() -> None:
    """Two identical records never equal one multiplicity-2 record.

    ``(x - 1)^2`` claimed as two separate multiplicity-1 records of the
    same factor reconstructs exactly, but the canonical irreducible
    factorization lists one record; repeating a factor key is rejected.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialFactorizationResult,
        PolynomialIrreducibleFactor,
    )

    factor = _univariate("x", {1: "1", 0: "-1"})
    source = _univariate("x", {2: "1", 1: "-2", 0: "1"})
    with polynomial_validation_error():
        PolynomialFactorizationResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialIrreducibleFactor(factor=factor, multiplicity=1),
                PolynomialIrreducibleFactor(factor=factor, multiplicity=1),
            ),
            reconstructed=source,
        )


def test_square_free_replay_rejects_repeated_part_keys() -> None:
    """The same part listed at two multiplicities is not a decomposition.

    ``(x - 1)^3`` claimed as ``(x - 1)`` at multiplicities 1 and 2
    reconstructs exactly, but the parts overlap, so the repeated key is
    rejected before any coprimality work.
    """

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    factor = _univariate("x", {1: "1", 0: "-1"})
    source = _univariate("x", {3: "1", 2: "-3", 1: "3", 0: "-1"})
    with polynomial_validation_error():
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(
                PolynomialSquareFreeFactor(factor=factor, multiplicity=1),
                PolynomialSquareFreeFactor(factor=factor, multiplicity=2),
            ),
            reconstructed=source,
        )
