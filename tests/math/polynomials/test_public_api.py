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

    from pydantic import ValidationError

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
    with pytest.raises(ValidationError, match="one variable over QQ"):
        PolynomialFactorizationResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=factors,
            reconstructed=source,
        )


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
        # The factorization contract is univariate, so a forged QQ[x, y]
        # source is rejected by the semantic-domain check before any replay
        # work; square-free decomposition admits several variables and must
        # still stop the monster at the first inexact division.
        expected = (
            "one variable over QQ"
            if "Factorization" in result_model.__name__
            else "must reconstruct"
        )
        with pytest.raises(ValidationError, match=expected):
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

    from pydantic import ValidationError

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials._models import (
        PolynomialSquareFreeDecompositionResult,
        PolynomialSquareFreeFactor,
    )

    variables = tuple(f"v{index}" for index in range(8))
    source, factor = _multivariate_binomial_box(variables, 64)
    with pytest.raises(ValidationError, match="must reconstruct"):
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

    from pydantic import ValidationError

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
    with pytest.raises(ValidationError, match="must reconstruct"):
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialSquareFreeFactor(factor=forged_factor, multiplicity=1),),
            reconstructed=source,
        )


def test_square_free_results_require_canonical_square_free_parts() -> None:
    """A product-correct but non-coprime grouping is not the decomposition.

    ``(x - 1) * ((x - 1)(y - 1))**2`` equals the retained source
    ``(x - 1)**3 * (y - 1)**2``, so every exact-division step would
    succeed; only the unique canonical monic square-free parts
    authenticate the claimed records.
    """

    from pydantic import ValidationError

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
    with pytest.raises(ValidationError, match="must reconstruct"):
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


def test_square_free_replay_rejects_inexact_multivariate_divisors() -> None:
    """A forged divisor with a tiny degree box is rejected without dividing.

    The claimed factor ``v_0 - sum(v_i^64)`` over eight variables leaves
    per-variable degree room ``64`` and ``1`` against the monomial source
    ``prod(v_i^64)``, so a division replay would pass any box bound while
    its inexact lexicographic long division builds huge partial sums.  The
    recomparison lane rejects the mismatched records typedly instead.
    """

    from pydantic import ValidationError

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
    with pytest.raises(ValidationError, match="must reconstruct"):
        PolynomialSquareFreeDecompositionResult(
            polynomial=source,
            coefficient=CanonicalRational(num="1", den="1"),
            factors=(PolynomialSquareFreeFactor(factor=factor, multiplicity=1),),
            reconstructed=source,
        )
