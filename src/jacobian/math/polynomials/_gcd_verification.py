"""Bounded exact checks for source-bound monic GCD and Bézout claims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._models import (
    PolynomialBezoutIdentity,
    PolynomialGcdResult,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_MAX_REPLAY_PRODUCTS = 1_000_000
_MAX_REPLAY_BITS = 262_144
_MAX_REPLAY_BIT_WORK = 1_000_000_000_000


def _resource() -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=(),
        code="polynomial.gcd_verification_budget",
        message="GCD relation replay exceeds its coefficient or bit-work envelope",
    )


@dataclass
class _Budget:
    remaining: int = _MAX_REPLAY_BIT_WORK

    def charge(self, operations: int, bits: int) -> None:
        work = operations * max(1, bits) ** 2
        if bits > _MAX_REPLAY_BITS or work > self.remaining:
            raise _resource()
        self.remaining -= work


def _bounded_shape(value: RationalPolynomial) -> bool:
    """Bound forged native containers and integers before making a parsing copy."""
    if type(value) is not RationalPolynomial:
        return False
    if (
        value.domain != "QQ"
        or type(value.variables) is not tuple
        or len(value.variables) != 1
        or type(value.variables[0]) is not str
        or len(value.variables[0]) > 32
        or type(value.polynomial) is not SparseRationalPolynomial
        or type(value.polynomial.terms) is not tuple
        or len(value.polynomial.terms) > MAX_POLYNOMIAL_TERMS
    ):
        return False
    for term in value.polynomial.terms:
        if (
            type(term) is not RationalPolynomialTerm
            or type(term.exponents) is not tuple
            or len(term.exponents) != 1
            or type(term.exponents[0]) is not int
            or not 0 <= term.exponents[0] <= MAX_POLYNOMIAL_EXPONENT
            or type(term.coefficient) is not CanonicalRational
        ):
            return False
        for component in (term.coefficient.num, term.coefficient.den):
            if type(component) is not int:
                return False
            if component.bit_length() > 4 * MAX_CANONICAL_RATIONAL_DIGITS:
                return False
    return True


def _primitive_integer_coefficients(
    polynomial: RationalPolynomial, budget: _Budget
) -> dict[int, int]:
    """Clear denominators after stripping scalar content, within the same QQ ideal."""
    terms = polynomial.polynomial.terms
    numerator_content, denominator_content = 0, 0
    for term in terms:
        coefficient = term.coefficient
        bits = max(coefficient.num.bit_length(), coefficient.den.bit_length())
        budget.charge(2, bits)
        numerator_content = gcd(numerator_content, coefficient.num)
        denominator_content = gcd(denominator_content, coefficient.den)
    ratios = [
        (
            term.coefficient.num // numerator_content,
            term.coefficient.den // denominator_content,
        )
        for term in terms
    ]
    denominator = 1
    for _, component in ratios:
        budget.charge(2, max(denominator.bit_length(), component.bit_length()))
        factor = component // gcd(denominator, component)
        budget.charge(1, denominator.bit_length() + factor.bit_length())
        denominator *= factor
    result = {}
    for term, (numerator, component) in zip(terms, ratios, strict=True):
        budget.charge(2, numerator.bit_length() + denominator.bit_length())
        result[term.exponents[0]] = numerator * (denominator // component)
    return result


def _divides(
    divisor: RationalPolynomial, source: RationalPolynomial, budget: _Budget
) -> bool:
    if not source.polynomial.terms:
        return True
    divisor_terms = divisor.polynomial.terms
    degree = divisor_terms[0].exponents[0]
    if len(divisor_terms) == 1:
        # The monic divisor x^degree needs neither coefficient arithmetic nor
        # dense allocation, including degree zero and maximum-width sources.
        return all(term.exponents[0] >= degree for term in source.polynomial.terms)
    source_degree = source.polynomial.terms[0].exponents[0]
    if source_degree < degree:
        return False
    remainder = _primitive_integer_coefficients(source, budget)
    integer_divisor = _primitive_integer_coefficients(divisor, budget)
    leading = integer_divisor[degree]
    divisor_bits = max(abs(value).bit_length() for value in integer_divisor.values())
    # The degree strictly drops, so there are at most source_degree-degree+1
    # steps. Pre-admit each actual sparse step against the shared total budget,
    # rather than refusing sparse one-step divisions using a dense worst case.
    # Every product/subtraction fits h(R)+h(B)+1 bits; support never exceeds
    # source_degree+1, and is allocated only after this step is charged.
    while remainder and (remainder_degree := max(remainder)) >= degree:
        remainder_bits = max(abs(value).bit_length() for value in remainder.values())
        bits = remainder_bits + divisor_bits + 1
        budget.charge(4 * (len(remainder) + len(integer_divisor)), bits)
        top = remainder[remainder_degree]
        shift = remainder_degree - degree
        remainder = {power: leading * value for power, value in remainder.items()}
        for power, value in integer_divisor.items():
            target = power + shift
            updated = remainder.get(target, 0) - top * value
            if updated:
                remainder[target] = updated
            else:
                remainder.pop(target, None)
    return not remainder


def _canonical_claim(
    claim: PolynomialGcdResult, budget: _Budget
) -> PolynomialGcdResult | None:
    if type(claim) is not PolynomialGcdResult:
        return None
    try:
        if (
            claim.normalization != "MONIC"
            or type(claim.bezout) is not PolynomialBezoutIdentity
        ):
            return None
        candidates = (
            claim.left,
            claim.right,
            claim.gcd,
            claim.bezout.left_multiplier,
            claim.bezout.right_multiplier,
        )
        if not all(_bounded_shape(value) for value in candidates):
            return None
        for value in candidates:
            for term in value.polynomial.terms:
                bits = max(
                    term.coefficient.num.bit_length(), term.coefficient.den.bit_length()
                )
                budget.charge(8, bits)
        # Bounded plain-field reparsing rejects forged native model copies.
        return PolynomialGcdResult.model_validate(claim.model_dump())
    except (AttributeError, ValidationError):
        return None


def _accumulate(
    coefficients: dict[int, Fraction], power: int, value: Fraction, budget: _Budget
) -> None:
    previous = coefficients.get(power)
    if previous is None:
        coefficients[power] = value
        return
    if previous == -value:
        del coefficients[power]
        return
    # Bound unreduced cross-products plus one carry bit before addition.
    bits = (
        max(
            previous.numerator.bit_length() + value.denominator.bit_length(),
            value.numerator.bit_length() + previous.denominator.bit_length(),
            previous.denominator.bit_length() + value.denominator.bit_length(),
        )
        + 1
    )
    budget.charge(8, bits)
    coefficients[power] = previous + value


def _identity_holds(claim: PolynomialGcdResult, budget: _Budget) -> bool:
    pairs = (
        (claim.bezout.left_multiplier, claim.left),
        (claim.bezout.right_multiplier, claim.right),
    )
    products = sum(len(a.polynomial.terms) * len(b.polynomial.terms) for a, b in pairs)
    if products > _MAX_REPLAY_PRODUCTS:
        raise _resource()
    budget.charge(products, 32)
    coefficients = {
        term.exponents[0]: -term.coefficient.as_fraction()
        for term in claim.gcd.polynomial.terms
    }
    # Stay sparse: a backend's dense common-denominator representation can be
    # much larger than these canonical coefficients, even for a zero product.
    # There are at most 2*MAX_POLYNOMIAL_EXPONENT+1 accumulator entries. Each
    # pair and addition is admitted before expansion, against one total budget.
    for left, right in pairs:
        if not left.polynomial.terms or not right.polynomial.terms:
            continue
        left_terms = [
            (term.exponents[0], term.coefficient.as_fraction())
            for term in left.polynomial.terms
        ]
        right_terms = [
            (term.exponents[0], term.coefficient.as_fraction())
            for term in right.polynomial.terms
        ]
        for left_power, a in left_terms:
            for right_power, b in right_terms:
                bits = max(
                    a.numerator.bit_length() + b.numerator.bit_length(),
                    a.denominator.bit_length() + b.denominator.bit_length(),
                )
                budget.charge(4, bits)
                _accumulate(coefficients, left_power + right_power, a * b, budget)
    return not coefficients


def verify_gcd_relation(claim: PolynomialGcdResult) -> bool:
    """Check monicity, both divisibilities, and the authored Bézout identity."""
    budget = _Budget()
    canonical = _canonical_claim(claim, budget)
    if canonical is None:
        return False
    left, right, divisor = canonical.left, canonical.right, canonical.gcd
    s, t = canonical.bezout.left_multiplier, canonical.bezout.right_multiplier
    if any(value.variables != left.variables for value in (right, divisor, s, t)):
        return False
    if (
        not divisor.polynomial.terms
        or divisor.polynomial.terms[0].coefficient.as_fraction() != 1
    ):
        return False
    if not _identity_holds(canonical, budget):
        return False
    return _divides(divisor, left, budget) and _divides(divisor, right, budget)
