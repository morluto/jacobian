"""Complete univariate division envelopes for the maintained classical kernel."""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import RationalPolynomial, RationalPolynomialTerm

_CANONICAL_LIMIT = 10**MAX_CANONICAL_RATIONAL_DIGITS
MAX_DIVISION_PRIVATE_BITS = 262_144
MAX_DIVISION_WORK = 1 << 37
MAX_DIVISION_ALLOCATION_BITS = 536_870_912
MAX_DIVISION_OUTPUT_CHARACTERS = 9_000_000


def _reject(reason: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code=f"polynomial.division.{reason}",
        message=message,
    )


@dataclass
class _Ledger:
    """Policy work proxy in squared 64-bit limbs, not a machine instruction count."""

    work: int = 0

    def charge(self, operations: int, bits: int = 1) -> None:
        request_checkpoint("during univariate division admission")
        if bits > MAX_DIVISION_PRIVATE_BITS:
            _reject(
                "intermediate_height",
                "division exceeds its private coefficient-bit envelope",
            )
        self.work += operations * max(1, (bits + 63) // 64) ** 2
        if self.work > MAX_DIVISION_WORK:
            _reject("work", "division exceeds its exact scalar-work envelope")

    def add(self, left: int, right: int) -> int:
        self.charge(1, max(left.bit_length(), right.bit_length()) + 1)
        return left + right

    def multiply(self, left: int, right: int) -> int:
        self.charge(1, left.bit_length() + right.bit_length())
        return left * right


def _source(
    polynomial: RationalPolynomial, ledger: _Ledger
) -> tuple[RationalPolynomialTerm, ...]:
    """Owner source budgets precede this bounded canonical-shape recognition."""
    terms = polynomial.polynomial.terms
    previous = 128
    for term in terms:
        if (
            type(term) is not RationalPolynomialTerm
            or type(term.exponents) is not tuple
            or len(term.exponents) != 1
            or type(term.exponents[0]) is not int
            or not 0 <= term.exponents[0] < previous
            or type(term.coefficient) is not CanonicalRational
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.division.source",
                message="division needs canonical ordered univariate terms",
            )
        previous = term.exponents[0]
        coefficient = term.coefficient
        if (
            type(coefficient.num) is not int
            or type(coefficient.den) is not int
            or not coefficient.num
            or coefficient.den <= 0
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.division.source",
                message="division needs nonzero canonical rational coefficients",
            )
        ledger.charge(
            1, abs(coefficient.num).bit_length() + coefficient.den.bit_length()
        )
        if gcd(coefficient.num, coefficient.den) != 1:
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.division.source",
                message="division needs reduced rational coefficients",
            )
    return terms


@dataclass(frozen=True)
class _IntegralSource:
    denominator: int
    coefficients: tuple[int, ...]
    height: int


def _clear(
    terms: tuple[RationalPolynomialTerm, ...], ledger: _Ledger
) -> _IntegralSource:
    denominator = 1
    for value in dict.fromkeys(term.coefficient.den for term in terms):
        ledger.charge(1, denominator.bit_length() + value.bit_length())
        denominator = ledger.multiply(denominator // gcd(denominator, value), value)
    coefficients = []
    for term in terms:
        ledger.charge(1, denominator.bit_length() + term.coefficient.den.bit_length())
        absolute = ledger.multiply(
            abs(term.coefficient.num), denominator // term.coefficient.den
        )
        coefficients.append(-absolute if term.coefficient.num < 0 else absolute)
    return _IntegralSource(
        denominator, tuple(coefficients), max(map(abs, coefficients), default=0)
    )


def _digits(value: int) -> int:
    return 1 if not value else (abs(value).bit_length() * 30103) // 100000 + 1


def _source_characters(terms: tuple[RationalPolynomialTerm, ...]) -> int:
    return 256 + sum(
        128 + _digits(t.coefficient.num) + _digits(t.coefficient.den) for t in terms
    )


def _component(value: int) -> None:
    if value >= _CANONICAL_LIMIT:
        _reject(
            "coefficient_height",
            "division quotient or remainder can exceed the canonical scalar carrier",
        )


def admit_univariate_division(
    left: RationalPolynomial, right: RationalPolynomial
) -> None:
    """Prove classical quotient/remainder, conversion and reconstruction bounds.

    For integral A=L*f, B=M*g, write b=|LC(B)|, H=max|A_i|,
    C=b+max(nonleading |B_i|), t=deg(f)-deg(g)+1. After j steps the
    unscaled remainder has denominator b**j and numerator height <=H*C**j.
    Thus Q has numerator <=M*H*C**(t-1), R <=H*C**t, and both denominators
    divide D=L*b**t. These are bounds, not assertions that all factors survive.
    The exact sparse self/monomial/degree branches retain sharper envelopes.
    """
    ledger = _Ledger()
    a_terms, b_terms = _source(left, ledger), _source(right, ledger)
    if not b_terms:
        raise AssertionError("owner must reject a zero divisor before growth admission")
    a = _clear(a_terms, ledger)
    b = a if left == right else _clear(b_terms, ledger)
    n = a_terms[0].exponents[0] if a_terms else 0
    m = b_terms[0].exponents[0]
    quotient_terms, remainder_terms = max(n - m + 1, 0), min(m, n + 1)
    qnum = qden = rnum = rden = 1
    dense_qnum = dense_qden = dense_rnum = dense_rden = 1
    if not a_terms or left == right:
        quotient_terms, remainder_terms = int(bool(a_terms)), 0
    elif n < m:
        quotient_terms, remainder_terms = 0, len(a_terms)
        rnum = max(abs(t.coefficient.num) for t in a_terms)
        rden = max(t.coefficient.den for t in a_terms)
        dense_rnum, dense_rden = a.height, a.denominator
    elif len(b_terms) == 1:
        quotient_terms = remainder_terms = 0
        divisor = b_terms[0].coefficient
        dense_qnum = ledger.multiply(a.height, divisor.den)
        dense_qden = ledger.multiply(a.denominator, abs(divisor.num))
        dense_rnum, dense_rden = a.height, a.denominator
        for term in a_terms:
            if term.exponents[0] < m:
                remainder_terms += 1
                rnum = max(rnum, abs(term.coefficient.num))
                rden = max(rden, term.coefficient.den)
                continue
            ledger.charge(
                1,
                sum(
                    abs(v).bit_length()
                    for v in (
                        term.coefficient.num,
                        term.coefficient.den,
                        divisor.num,
                        divisor.den,
                    )
                ),
            )
            coefficient = Fraction(
                term.coefficient.num, term.coefficient.den
            ) / Fraction(divisor.num, divisor.den)
            quotient_terms += 1
            qnum, qden = (
                max(qnum, abs(coefficient.numerator)),
                max(qden, coefficient.denominator),
            )
    else:
        steps = n - m + 1
        leading = abs(b.coefficients[0])
        growth = ledger.add(leading, max(map(abs, b.coefficients[1:]), default=0))
        height, denominator = a.height, a.denominator
        for _ in range(steps):
            qnum = max(qnum, ledger.multiply(b.denominator, height))
            denominator = ledger.multiply(denominator, leading)
            height = ledger.multiply(height, growth)
            for value in (qnum, denominator, height):
                _component(value)
        qden = rden = denominator
        rnum = height
    for value in (qnum, qden, rnum, rden):
        _component(value)
    # Ring->Poly conversion clears divisors of D; reconstruction Q*g+R uses
    # common denominator D*M. Products before gcd reduction can double this
    # bit width. Its convolution has at most deg(g)+1 contributions per slot.
    q_global_num = max(qnum, dense_qnum)
    q_global_den = max(qden, dense_qden)
    r_global_num = max(rnum, dense_rnum)
    r_global_den = max(rden, dense_rden)
    common_bits = (
        max(q_global_den.bit_length(), r_global_den.bit_length())
        + b.denominator.bit_length()
    )
    reconstruction_bits = (
        max(
            q_global_num.bit_length() + b.height.bit_length(),
            r_global_num.bit_length() + b.denominator.bit_length(),
        )
        + (n + 2).bit_length()
    )
    source_bits = max(
        a.denominator.bit_length(),
        b.denominator.bit_length(),
        a.height.bit_length(),
        b.height.bit_length(),
    )
    private_bits = (
        2 * max(common_bits, reconstruction_bits, source_bits)
        + (n + 2).bit_length()
        + 4
    )
    steps = max(n - m + 1, 0)
    arithmetic = (
        steps * (1 + 2 * len(b_terms)) + 2 * quotient_terms * len(b_terms) + 2 * (n + 1)
    )
    conversion = 32 * (n + m + 2)
    ledger.charge(arithmetic + conversion, private_bits)
    ledger.charge((n + 1) ** 2 + (m + 1) ** 2)
    # Reserve sixteen live-equivalent dense coefficient vectors for source,
    # conversion, convolution, replay and canonical construction. This bounds
    # exact coefficient storage, not process RSS or Python object overhead.
    allocation = 16 * (max(n, m) + 1) * private_bits
    if allocation > MAX_DIVISION_ALLOCATION_BITS:
        _reject(
            "allocation",
            "division exceeds its aggregate exact-coefficient storage envelope",
        )
    characters = (
        2 * _source_characters(a_terms)
        + _source_characters(b_terms)
        + quotient_terms * (128 + _digits(qnum) + _digits(qden))
        + remainder_terms * (128 + _digits(rnum) + _digits(rden))
        + 1024
    )
    if characters > MAX_DIVISION_OUTPUT_CHARACTERS:
        _reject(
            "output_size", "division exceeds its complete canonical output envelope"
        )
