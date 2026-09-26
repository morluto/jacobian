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


def test_native_expression_normalizer_is_public_and_typed() -> None:
    from jacobian.math.polynomials._expression_normalize import (
        PolynomialExpressionSource,
    )

    source = PolynomialExpressionSource.model_validate(
        {
            "coefficient_domain": "ZZ",
            "variables": ["x"],
            "expression": {
                "kind": "POWER",
                "base": {
                    "kind": "ADD",
                    "operands": [
                        {"kind": "VARIABLE", "name": "x"},
                        {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                    ],
                },
                "exponent": 2,
            },
        }
    )
    result = polynomials.normalize_polynomial_expression(source)
    assert tuple(term.exponents for term in result.polynomial.polynomial.terms) == (
        (2,),
        (1,),
        (0,),
    )


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
        "PolynomialExpressionSource",
        "QuarticCubicResolventResult",
        "RationalDiscreteAntiderivativeResult",
        "RationalLaurentPolynomial",
        "RationalLaurentPolynomialTerm",
        "compute_quartic_cubic_resolvent",
        "cyclotomic",
        "derivative",
        "discriminant",
        "divide",
        "elementary_symmetric_family",
        "evaluate",
        "factorization",
        "gcdex",
        "groebner_basis",
        "hermite_reduction",
        "ideal_containment",
        "ideal_equality",
        "ideal_membership_certificate",
        "ideal_normal_form",
        "integer_polynomial_compose",
        "integer_polynomial_content",
        "integer_polynomial_evaluate",
        "integer_polynomial_gcd",
        "integer_polynomial_primitive_part",
        "integer_polynomial_shift",
        "integral",
        "mahler_measure",
        "multiply",
        "normalize_polynomial_expression",
        "partial_fractions",
        "polynomial_discriminant",
        "polynomial_factorization",
        "polynomial_gcd",
        "polynomial_groebner_basis",
        "polynomial_resultant",
        "polynomial_square_free_decomposition",
        "quadratic_root_profile",
        "rational_discrete_antiderivative",
        "rational_laurent_multiply",
        "rational_partial_fraction_decomposition",
        "rational_polynomial_derivative",
        "rational_polynomial_division",
        "rational_polynomial_evaluate",
        "rational_polynomial_integral",
        "reciprocal_profile",
        "resultant",
        "square_free_decomposition",
        "verify_hermite_reduction",
        "verify_polynomial_discriminant",
        "verify_polynomial_factorization",
        "verify_polynomial_gcd",
        "verify_polynomial_resultant",
        "verify_polynomial_square_free_decomposition",
    )
    assert tuple(polynomials.__all__) == expected
    assert len(polynomials.__all__) == len(set(polynomials.__all__))
    assert all(
        not name.startswith("_") and hasattr(polynomials, name) for name in expected
    )


def test_native_laurent_api_preserves_signed_support_and_zero_parent() -> None:
    from jacobian._exact import CanonicalRational

    left = polynomials.RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            polynomials.RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1), exponents=(1,)
            ),
            polynomials.RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1), exponents=(-1,)
            ),
        ),
    )
    right = polynomials.RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            polynomials.RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=1), exponents=(1,)
            ),
            polynomials.RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=-1, den=1), exponents=(-1,)
            ),
        ),
    )

    result = polynomials.rational_laurent_multiply(left, right)

    assert result.variables == ("x",)
    assert tuple(term.exponents for term in result.terms) == ((2,), (-2,))


def test_native_discrete_antiderivative_api_returns_typed_result() -> None:
    source = _univariate("k", {2: 1})
    result = polynomials.rational_discrete_antiderivative(
        source,
        "k",
    )
    assert isinstance(
        result,
        polynomials.RationalDiscreteAntiderivativeResult,
    )
    assert result.reconstructed_difference == source


def _univariate(variable: str, terms: dict[int, int]) -> Any:
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
                        num=value,
                        den=1,
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

    source = _univariate("x", {4: 1, 2: -2, 0: 1})
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

    result = polynomial_factorization(_univariate("x", {3: 1, 0: -1}))
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
            reconstructed=_univariate("y", {3: 1, 0: -1}),
        )
    foreign = PolynomialIrreducibleFactor(
        factor=_univariate("y", {1: 1, 0: -1}), multiplicity=1
    )
    with pytest.raises(ValidationError):
        PolynomialFactorizationResult(
            polynomial=result.polynomial,
            coefficient=result.coefficient,
            factors=(foreign,),
            reconstructed=result.reconstructed,
        )


def _rebuild_native_factorization(
    source: Poly,
    coefficient: Any,
    factors: tuple[tuple[Poly, int], ...],
) -> Poly:
    """Rebuild a native factorization claim without trusting `reconstructed`."""
    rebuilt = Poly(coefficient, *source.gens, domain=source.domain)
    for factor, multiplicity in factors:
        assert factor.is_monic
        assert multiplicity >= 1
        rebuilt *= factor**multiplicity
    return Poly(rebuilt.as_expr(), *source.gens, domain=source.domain)


def test_native_factorization_rebuilds_source_independently() -> None:
    """Exact postcondition: coeff * prod(f_i**m_i) == source, proved by hand.

    Covers a repeated non-monic case with unequal multiplicities
    (6*(x-1)^2*(x+2)), a non-monic case with an irreducible quadratic
    (2*(x^2+1)*(x+1)), and an equal-multiplicity case ((x^2-1)^2).
    """
    x = symbols("x")
    cases = (
        Poly(6 * (x - 1) ** 2 * (x + 2), x, domain="QQ"),
        Poly(2 * (x**2 + 1) * (x + 1), x, domain="QQ"),
        Poly((x**2 - 1) ** 2, x, domain="QQ"),
    )
    for source in cases:
        assert source.LC() != 1 or source == cases[2]
        coefficient, factors, _ = polynomials.factorization(source)
        rebuilt = _rebuild_native_factorization(source, coefficient, factors)
        assert rebuilt == source
    coefficient, factors, _ = polynomials.factorization(cases[0])
    assert coefficient == 6
    assert {factor.as_expr(): multiplicity for factor, multiplicity in factors} == {
        (x - 1): 2,
        (x + 2): 1,
    }
    coefficient, factors, _ = polynomials.factorization(cases[2])
    assert coefficient == 1
    assert {factor.as_expr(): multiplicity for factor, multiplicity in factors} == {
        (x - 1): 2,
        (x + 1): 2,
    }
    quadratic = next(
        factor
        for factor, _ in polynomials.factorization(cases[1])[1]
        if Poly(factor.as_expr(), x, domain="QQ").degree() == 2
    )
    assert quadratic.as_expr() == x**2 + 1
    # x^2 + 1 is irreducible over QQ: its discriminant is not a rational square.
    assert polynomials.discriminant(Poly(x**2 + 1, x, domain="QQ"), x) == -4
    assert Poly(x**2 + 1, x, domain="QQ").is_irreducible


def test_native_factorization_forged_claim_fails_independent_rebuild() -> None:
    """A weakened claim (dropped multiplicity, scaled unit) cannot rebuild."""
    from jacobian.math.polynomials.operations import (
        polynomial_factorization,
        verify_polynomial_factorization,
    )

    x = symbols("x")
    source = Poly(6 * (x - 1) ** 2 * (x + 2), x, domain="QQ")
    coefficient, factors, _ = polynomials.factorization(source)
    weakened = tuple(
        (factor, multiplicity - 1 if multiplicity > 1 else multiplicity)
        for factor, multiplicity in factors
    )
    assert _rebuild_native_factorization(source, coefficient, weakened) != source
    assert _rebuild_native_factorization(source, 2 * coefficient, factors) != source

    typed = polynomial_factorization(_univariate("x", {3: 1, 1: -3, 0: 2}))
    assert verify_polynomial_factorization(typed)
    tampered_factors = tuple(
        type(record)(factor=record.factor, multiplicity=record.multiplicity + 1)
        for record in typed.factors
    )
    forged = typed.model_copy(update={"factors": tampered_factors})
    assert not verify_polynomial_factorization(forged)


def test_native_gcd_bezout_divisibility_both_orders_and_degenerate() -> None:
    """GCD postconditions: Bezout identity, divisibility, monic, symmetry."""
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials.operations import (
        polynomial_gcd,
        verify_polynomial_gcd,
    )

    left = _univariate("x", {3: 1, 1: -3, 0: 2})  # (x-1)^2 (x+2)
    right = _univariate("x", {3: 1, 2: 3, 0: -4})  # (x-1)(x+2)^2
    forward = polynomial_gcd(left, right)
    backward = polynomial_gcd(right, left)
    assert forward.gcd == backward.gcd
    assert verify_polynomial_gcd(forward)
    expected_gcd = _univariate("x", {2: 1, 1: 1, 0: -2})  # (x-1)(x+2)
    assert forward.gcd == expected_gcd
    for claim in (forward, backward):
        left_sym = rational_polynomial_to_sympy(claim.left)
        right_sym = rational_polynomial_to_sympy(claim.right)
        gcd_sym = rational_polynomial_to_sympy(claim.gcd)
        multiplier_sym = rational_polynomial_to_sympy(claim.bezout.left_multiplier)
        other_sym = rational_polynomial_to_sympy(claim.bezout.right_multiplier)
        assert gcd_sym.is_monic
        assert left_sym * multiplier_sym + right_sym * other_sym == gcd_sym
        for operand in (left_sym, right_sym):
            _, remainder = operand.div(gcd_sym)
            assert remainder.is_zero
    # Degenerate: gcd(f, 0) is the monic associate of f, in both orders.
    zero = _univariate("x", {})
    monic_source = _univariate("x", {2: 1, 0: -1})
    assert polynomial_gcd(monic_source, zero).gcd == monic_source
    assert polynomial_gcd(zero, monic_source).gcd == monic_source


def test_native_gcd_forged_claim_fails_bezout_and_verify() -> None:
    """Claiming an operand as the GCD breaks divisibility and verification."""
    from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
    from jacobian.math.polynomials.operations import (
        polynomial_gcd,
        verify_polynomial_gcd,
    )

    left = _univariate("x", {3: 1, 1: -3, 0: 2})
    right = _univariate("x", {3: 1, 2: 3, 0: -4})
    result = polynomial_gcd(left, right)
    forged = result.model_copy(update={"gcd": result.left})
    assert not verify_polynomial_gcd(forged)
    claimed = rational_polynomial_to_sympy(forged.gcd)
    target = rational_polynomial_to_sympy(forged.right)
    _, remainder = target.div(claimed)
    assert not remainder.is_zero


def _native_sylvester_determinant(left: Poly, right: Poly, generator: Any) -> Any:
    """Independent Sylvester-determinant oracle for the native resultant."""
    from sympy import Matrix

    left_coeffs = left.all_coeffs()
    right_coeffs = right.all_coeffs()
    m = left.degree(generator)
    n = right.degree(generator)
    assert isinstance(m, int) and isinstance(n, int)
    if m == 0 and n == 0:
        return left_coeffs[0] * 0 + 1
    size = m + n
    rows: list[list[Any]] = []
    for index in range(n):
        rows.append([0] * index + list(left_coeffs) + [0] * (n - 1 - index))
    for index in range(m):
        rows.append([0] * index + list(right_coeffs) + [0] * (m - 1 - index))
    assert all(len(row) == size for row in rows)
    return Matrix(rows).det()


def test_native_resultant_matches_sylvester_in_both_orders() -> None:
    """Res(f,g) equals the Sylvester determinant; swap law holds exactly."""
    x = symbols("x")
    pairs = (
        (Poly(x**2 - 3 * x + 2, x, domain="QQ"), Poly(2 * x + 1, x, domain="QQ")),
        (Poly(x + 2, x, domain="QQ"), Poly(x**3 + 1, x, domain="QQ")),
    )
    for left, right in pairs:
        forward = polynomials.resultant(left, right, x)
        reverse = polynomials.resultant(right, left, x)
        assert forward == _native_sylvester_determinant(left, right, x)
        assert reverse == _native_sylvester_determinant(right, left, x)
        m = left.degree(x)
        n = right.degree(x)
        assert isinstance(m, int) and isinstance(n, int)
        assert forward == (-1) ** (m * n) * reverse
    # Degenerate: resultant against a nonzero constant is a pure power.
    base = Poly(x**2 + 1, x, domain="QQ")
    constant = Poly(5, x, domain="QQ")
    assert polynomials.resultant(base, constant, x) == 5 ** base.degree(x)
    assert polynomials.resultant(constant, base, x) == 5 ** base.degree(x)
    assert polynomials.resultant(base, Poly(0, x, domain="QQ"), x) == 0


def test_native_resultant_forged_sign_fails_sylvester_oracle() -> None:
    """Negating an odd-degree-pair resultant contradicts the Sylvester value."""
    x = symbols("x")
    left = Poly(x + 2, x, domain="QQ")
    right = Poly(x**3 + 1, x, domain="QQ")
    honest = polynomials.resultant(left, right, x)
    assert honest == -7
    assert -honest != _native_sylvester_determinant(left, right, x)

    from jacobian.math.polynomials._models import PolynomialScalarValue
    from jacobian.math.polynomials.operations import (
        polynomial_resultant,
        verify_polynomial_resultant,
    )

    typed_left = _univariate("x", {1: 1, 0: 2})
    typed_right = _univariate("x", {3: 1, 0: 1})
    result = polynomial_resultant(typed_left, typed_right, "x")
    assert verify_polynomial_resultant(result)
    assert isinstance(result.resultant, PolynomialScalarValue)
    forged_value = PolynomialScalarValue(
        value=type(result.resultant.value)(
            num=-result.resultant.value.num, den=result.resultant.value.den
        )
    )
    assert not verify_polynomial_resultant(
        result.model_copy(update={"resultant": forged_value})
    )


def test_native_discriminant_satisfies_resultant_relation() -> None:
    """disc(f) = (-1)^{n(n-1)/2} Res(f,f')/lc(f), checked independently."""
    x = symbols("x")
    cases = (
        Poly(x**3 + x + 1, x, domain="QQ"),  # discriminant -31
        Poly(x**2 + x + 1, x, domain="QQ"),  # discriminant -3
    )
    for source in cases:
        degree = source.degree(x)
        assert isinstance(degree, int) and degree >= 2
        discriminant = polynomials.discriminant(source, x)
        derivative = polynomials.derivative(source)
        resultant = polynomials.resultant(source, derivative, x)
        assert resultant != 0
        assert (
            discriminant
            == (-1) ** (degree * (degree - 1) // 2) * resultant / source.LC()
        )
    assert polynomials.discriminant(cases[0], x) == -31
    assert polynomials.discriminant(cases[1], x) == -3
    # Degenerate: linear polynomials have discriminant one; repeated roots give zero.
    assert polynomials.discriminant(Poly(3 * x + 2, x, domain="QQ"), x) == 1
    assert (
        polynomials.discriminant(Poly((x - 1) ** 2 * (x + 2), x, domain="QQ"), x) == 0
    )


def test_native_discriminant_forged_claim_fails_relation_and_verify() -> None:
    """A negated discriminant contradicts the resultant relation."""
    from jacobian.math.polynomials._models import PolynomialScalarValue
    from jacobian.math.polynomials.operations import (
        polynomial_discriminant,
        verify_polynomial_discriminant,
    )

    x = symbols("x")
    source = Poly(x**3 + x + 1, x, domain="QQ")
    derivative = polynomials.derivative(source)
    resultant = polynomials.resultant(source, derivative, x)
    assert polynomials.discriminant(source, x) == -31
    assert -polynomials.discriminant(source, x) != (-1) ** 3 * resultant / source.LC()

    typed = polynomial_discriminant(_univariate("x", {3: 1, 1: 1, 0: 1}), "x")
    assert verify_polynomial_discriminant(typed)
    assert isinstance(typed.discriminant, PolynomialScalarValue)
    forged_value = PolynomialScalarValue(
        value=type(typed.discriminant.value)(
            num=-typed.discriminant.value.num, den=typed.discriminant.value.den
        )
    )
    assert not verify_polynomial_discriminant(
        typed.model_copy(update={"discriminant": forged_value})
    )


def test_native_square_free_rebuild_coprimality_and_degenerate() -> None:
    """Square-free postconditions: rebuild, square-free factors, coprimality."""
    x = symbols("x")
    source = Poly(12 * (x - 1) ** 2 * (x + 2) ** 3, x, domain="QQ")
    coefficient, factors, _ = polynomials.square_free_decomposition(source)
    assert _rebuild_native_factorization(source, coefficient, factors) == source
    assert coefficient == 12
    assert sorted(multiplicity for _, multiplicity in factors) == [2, 3]
    assert len({multiplicity for _, multiplicity in factors}) == len(factors)
    for index, (factor, _) in enumerate(factors):
        _, remainder = factor.div(polynomials.derivative(factor).gcd(factor))
        assert remainder.is_zero  # factor's square-free kernel divides it
        assert factor.gcd(polynomials.derivative(factor)).degree(x) == 0
        for other, _ in factors[index + 1 :]:
            assert factor.gcd(other).degree(x) == 0
    # Degenerate: a square-free input yields one multiplicity-one factor.
    clean = Poly(x**2 - 1, x, domain="QQ")
    clean_coefficient, clean_factors, _ = polynomials.square_free_decomposition(clean)
    assert (clean_coefficient, clean_factors) == (1, ((clean, 1),))
    assert (
        _rebuild_native_factorization(clean, clean_coefficient, clean_factors) == clean
    )


def test_native_square_free_forged_claim_fails_rebuild_and_verify() -> None:
    """Merging two square-free factors into one multiplicity breaks the claim."""
    from jacobian.math.polynomials.operations import (
        polynomial_square_free_decomposition,
        verify_polynomial_square_free_decomposition,
    )

    x = symbols("x")
    source = Poly(12 * (x - 1) ** 2 * (x + 2) ** 3, x, domain="QQ")
    coefficient, factors, _ = polynomials.square_free_decomposition(source)
    merged = factors[0][0] * factors[1][0]
    forged_factors = ((Poly(merged.as_expr(), x, domain="QQ").monic(), 2),)
    assert _rebuild_native_factorization(source, coefficient, forged_factors) != source

    typed = polynomial_square_free_decomposition(_univariate("x", {1: 1, 0: 1}))
    assert verify_polynomial_square_free_decomposition(typed)
    if len(typed.factors) >= 1:
        tampered = tuple(
            type(record)(factor=record.factor, multiplicity=record.multiplicity + 1)
            if index == 0
            else record
            for index, record in enumerate(typed.factors)
        )
        assert not verify_polynomial_square_free_decomposition(
            typed.model_copy(update={"factors": tampered})
        )
