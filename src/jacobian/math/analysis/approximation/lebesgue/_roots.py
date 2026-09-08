"""Exact critical points and their images, with admitted interval refinement."""

from dataclasses import dataclass
from fractions import Fraction

from sympy import Poly, Rational, Symbol

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
    _compare_admitted_real_algebraic,
)


@dataclass(frozen=True)
class Root:
    value: RealAlgebraicValue
    interval: RationalIsolatingInterval


def interval(lower: Fraction, upper: Fraction) -> RationalIsolatingInterval:
    return RationalIsolatingInterval(
        lower=CanonicalRational.from_fraction(lower),
        upper=CanonicalRational.from_fraction(upper),
        interval_type="SINGLETON" if lower == upper else "OPEN",
    )


def rational(value: Fraction) -> Root:
    return Root(
        RealAlgebraicValue._from_admitted_polynomial(
            polynomial=(value.denominator, -value.numerator), real_root_index=0
        ),
        interval(value, value),
    )


def compare(left: Root, right: Root) -> int:
    request_checkpoint("during exact Lebesgue extrema comparison")
    order = _compare_admitted_real_algebraic(
        left.value, right.value, left.interval, right.interval
    )
    return {"LT": -1, "EQ": 0, "GT": 1}[order]


def evaluate(coefficients: tuple[Fraction, ...], point: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


def image_interval(
    coefficients: tuple[Fraction, ...], lower: Fraction, upper: Fraction
) -> tuple[Fraction, Fraction]:
    a = b = Fraction(0)
    for coefficient in reversed(coefficients):
        products = (a * lower, a * upper, b * lower, b * upper)
        a, b = min(products) + coefficient, max(products) + coefficient
    return a, b


def factor_roots(polynomial: Poly, bits: int) -> tuple[tuple[Root, Poly, int], ...]:
    result = []
    for factor, multiplicity in polynomial.factor_list()[1]:
        request_checkpoint("during exact Lebesgue root isolation")
        factor = factor.primitive()[1]
        if factor.LC() < 0:
            factor = -factor
        encoded = tuple(int(c) for c in factor.all_coeffs())
        for index, ((a, b), _) in enumerate(factor.intervals(eps=Rational(1, 2**bits))):
            result.append(
                (
                    Root(
                        RealAlgebraicValue._from_admitted_polynomial(
                            polynomial=encoded, real_root_index=index
                        ),
                        interval(Fraction(a), Fraction(b)),
                    ),
                    factor,
                    int(multiplicity),
                )
            )
    return tuple(result)


def refine(root: Root, polynomial: Poly, bits: int) -> Root:
    a, b = polynomial.intervals(eps=Rational(1, 2**bits))[root.value.real_root_index][0]
    return Root(root.value, interval(Fraction(a), Fraction(b)))


def critical_points(
    coefficients: tuple[Fraction, ...],
    lower: Fraction,
    upper: Fraction,
    refinement_bits: int,
) -> tuple[tuple[Root, int, Root], ...]:
    x, y = Symbol("x"), Symbol("y")
    polynomial = Poly.from_dict(
        {
            (i,): Rational(c.numerator, c.denominator)
            for i, c in enumerate(coefficients)
        },
        x,
        domain="QQ",
    )
    denominator, integral = polynomial.clear_denoms(convert=True)
    derivative = integral.diff()
    roots = factor_roots(derivative, 8)
    selected = [
        (root, factor, multiplicity)
        for root, factor, multiplicity in roots
        if compare(root, rational(lower)) >= 0 and compare(root, rational(upper)) <= 0
    ]
    if not selected:
        return ()
    # The resultant contains every critical image, including possible images
    # of nonreal critical points. Interval association selects the actual image.
    images: tuple[tuple[Root, Poly, int], ...] = ()
    if any(len(root.value.polynomial) > 2 for root, _, _ in selected):
        request_checkpoint("before exact Lebesgue critical-value resultant")
        resultant = Poly(
            derivative.resultant(Poly(denominator * y - integral.as_expr(), x)),
            y,
            domain="ZZ",
        )
        images = factor_roots(resultant, 8)
    result = []
    for root, factor, multiplicity in selected:
        if len(root.value.polynomial) == 2:
            point = -Fraction(root.value.polynomial[1], root.value.polynomial[0])
            result.append((root, multiplicity, rational(evaluate(coefficients, point))))
            continue
        bits = 8
        candidates = images
        while True:
            request_checkpoint("during exact Lebesgue critical-value association")
            a, b = image_interval(
                coefficients,
                max(lower, root.interval.lower.as_fraction()),
                min(upper, root.interval.upper.as_fraction()),
            )
            candidates = tuple(
                entry
                for entry in candidates
                if entry[0].interval.lower.as_fraction() <= b
                and entry[0].interval.upper.as_fraction() >= a
            )
            if len(candidates) == 1:
                result.append((root, multiplicity, candidates[0][0]))
                break
            if not candidates or bits == refinement_bits:
                raise ArithmeticError(
                    "admitted critical-value separation did not resolve an image"
                )
            bits = min(2 * bits, refinement_bits)
            root = refine(root, factor, bits)
            candidates = tuple(
                (refine(image, image_factor, bits), image_factor, exponent)
                for image, image_factor, exponent in candidates
            )
    return tuple(result)
