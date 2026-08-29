"""Exact arithmetic on Python integers and fractions."""

from collections.abc import Iterable
from fractions import Fraction
from math import ceil, floor, gcd, isqrt, lcm
from typing import SupportsIndex

from jacobian._exact import CanonicalInteger
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_models import ExtendedGcdResult
from jacobian.math.number_theory._integer_models import BooleanResult
from jacobian.math.number_theory.arithmetic._models import (
    _MAX_BASE,
    _MAX_NTH_ROOT_DEGREE,
    MAX_BASE_DIGITS,
)
from jacobian.math.number_theory.arithmetic._multiplicative_forms import (
    MAX_K_VALUE,
    KFreeDecompositionResult,
    NormalizedQuadraticRadicalResult,
    PerfectPowerProfileResult,
    PrimeExponentRow,
    SquarefreeDecompositionResult,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue

__all__ = [
    "absolute_value",
    "aliquot_sum",
    "are_coprime",
    "base_digits",
    "ceiling_rational",
    "continued_fraction",
    "decimal_digit_count",
    "decimal_digit_sum",
    "difference_rationals",
    "divides",
    "divisor_count",
    "divisor_sum",
    "equal_rationals",
    "extended_gcd",
    "floor_rational",
    "integer_gcd",
    "integer_lcm",
    "integerize_rational_vector",
    "is_abundant",
    "is_deficient",
    "is_even",
    "is_odd",
    "is_perfect",
    "is_square",
    "k_free_decomposition",
    "less_than_rationals",
    "maximum_rational",
    "minimum_rational",
    "negate_rational",
    "normalized_quadratic_radical",
    "nth_root",
    "perfect_power_profile",
    "prime_valuation",
    "primitive_integer_vector",
    "product_rationals",
    "quotient",
    "rational_absolute_value",
    "reciprocal",
    "sign",
    "squarefree_decomposition",
    "sum_rationals",
]


def _as_python_integer(value: SupportsIndex | CanonicalInteger | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    if isinstance(value, str):
        return parse_canonical_integer(value)
    return value.__index__()


def _factorize_abs(value: int) -> list[tuple[int, int]]:
    """Return the complete prime factorization of |value| in sorted order."""

    from sympy import factorint

    return sorted(factorint(abs(value)).items())


def perfect_power_profile(value: IntegerValue) -> PerfectPowerProfileResult:
    """Compute the maximal perfect-power profile of one integer value."""

    integer = _as_python_integer(value)

    if integer == 0:
        return PerfectPowerProfileResult(kind="ZERO")
    if integer == 1:
        return PerfectPowerProfileResult(kind="POSITIVE_UNIT")
    if integer == -1:
        return PerfectPowerProfileResult(kind="NEGATIVE_UNIT")

    prime_exps = _factorize_abs(integer)
    exponent = prime_exps[0][1]
    for _, prime_exponent in prime_exps[1:]:
        exponent = gcd(exponent, prime_exponent)

    if integer < 0:
        while exponent % 2 == 0 and exponent > 1:
            exponent //= 2

    from sympy import integer_nthroot

    root, _ = integer_nthroot(abs(integer), exponent)
    base = -root if integer < 0 else root
    assert base**exponent == integer

    return PerfectPowerProfileResult(
        kind="NONUNIT",
        base=format_canonical_integer(base),
        exponent=exponent,
        is_nontrivial_perfect_power=exponent > 1,
        factors=tuple(
            PrimeExponentRow(
                prime=format_canonical_integer(prime),
                power=prime_exponent,
            )
            for prime, prime_exponent in prime_exps
        ),
        reconstruction=format_canonical_integer(integer),
    )


def _require_k(k: int) -> None:
    if not 2 <= k <= MAX_K_VALUE:
        raise OperationDomainValidationError(
            location=("k",),
            code="arithmetic.k_out_of_range",
            message=f"k must be between 2 and {MAX_K_VALUE}",
        )


def k_free_decomposition(value: IntegerValue, k: int) -> KFreeDecompositionResult:
    """Compute the unique decomposition n = a^k * c for one integer value."""

    _require_k(k)
    integer = _as_python_integer(value)

    if integer == 0:
        return KFreeDecompositionResult(kind="ZERO")
    if integer == 1 or integer == -1:
        return KFreeDecompositionResult(kind="UNIT")

    prime_exps = _factorize_abs(integer)
    base_value = 1
    cofactor_sign = 1 if integer > 0 else -1
    cofactor_abs = 1
    rows: list[PrimeExponentRow] = []

    for prime, exponent in prime_exps:
        quotient_value, remainder = divmod(exponent, k)
        if quotient_value > 0:
            base_value *= prime**quotient_value
        if remainder > 0:
            cofactor_abs *= prime**remainder
        rows.append(
            PrimeExponentRow(
                prime=format_canonical_integer(prime),
                power=exponent,
            )
        )

    cofactor = cofactor_sign * cofactor_abs
    assert base_value**k * cofactor == integer

    return KFreeDecompositionResult(
        kind="NONUNIT",
        base=format_canonical_integer(base_value),
        cofactor=format_canonical_integer(cofactor),
        factors=tuple(rows),
        reconstruction=format_canonical_integer(integer),
    )


def squarefree_decomposition(value: IntegerValue) -> SquarefreeDecompositionResult:
    """Compute the unique decomposition n = s^2 * d for one integer value."""

    integer = _as_python_integer(value)

    if integer == 0:
        return SquarefreeDecompositionResult(kind="ZERO")
    if integer == 1 or integer == -1:
        return SquarefreeDecompositionResult(kind="UNIT")

    prime_exps = _factorize_abs(integer)
    square_factor = 1
    squarefree_sign = 1 if integer > 0 else -1
    squarefree_abs = 1
    rows: list[PrimeExponentRow] = []

    for prime, exponent in prime_exps:
        quotient_value, remainder = divmod(exponent, 2)
        if quotient_value > 0:
            square_factor *= prime**quotient_value
        if remainder > 0:
            squarefree_abs *= prime
        rows.append(
            PrimeExponentRow(
                prime=format_canonical_integer(prime),
                power=exponent,
            )
        )

    squarefree_part = squarefree_sign * squarefree_abs
    assert square_factor**2 * squarefree_part == integer

    return SquarefreeDecompositionResult(
        kind="NONUNIT",
        square_factor=format_canonical_integer(square_factor),
        squarefree_part=format_canonical_integer(squarefree_part),
        factors=tuple(rows),
        reconstruction=format_canonical_integer(integer),
    )


def normalized_quadratic_radical(
    value: IntegerValue,
) -> NormalizedQuadraticRadicalResult:
    """Normalize the positive square root of one nonnegative integer value."""

    integer = _as_python_integer(value)
    if integer < 0:
        raise OperationDomainValidationError(
            location=("value",),
            code="arithmetic.value_must_be_nonnegative",
            message="value must be nonnegative",
        )

    if integer == 0:
        return NormalizedQuadraticRadicalResult(
            kind="ZERO", coefficient="0", radicand="1", reconstruction="0"
        )
    if integer == 1:
        return NormalizedQuadraticRadicalResult(
            kind="RATIONAL_INTEGER", coefficient="1", radicand="1", reconstruction="1"
        )

    prime_exps = _factorize_abs(integer)
    coefficient = 1
    radicand = 1
    for prime, exponent in prime_exps:
        quotient_value, remainder = divmod(exponent, 2)
        if quotient_value > 0:
            coefficient *= prime**quotient_value
        if remainder > 0:
            radicand *= prime

    assert coefficient**2 * radicand == integer
    return NormalizedQuadraticRadicalResult(
        kind="RATIONAL_INTEGER" if radicand == 1 else "IRRATIONAL_QUADRATIC",
        coefficient=format_canonical_integer(coefficient),
        radicand=format_canonical_integer(radicand),
        reconstruction=format_canonical_integer(integer),
    )


def integer_gcd(
    left: SupportsIndex | CanonicalInteger | IntegerValue,
    right: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the nonnegative greatest common divisor of two integers."""

    return IntegerValue(
        value=format_canonical_integer(
            gcd(_as_python_integer(left), _as_python_integer(right))
        )
    )


def integer_lcm(
    left: SupportsIndex | CanonicalInteger | IntegerValue,
    right: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the nonnegative least common multiple of two integers."""

    return IntegerValue(
        value=format_canonical_integer(
            lcm(_as_python_integer(left), _as_python_integer(right))
        )
    )


def extended_gcd(
    left: SupportsIndex | CanonicalInteger | IntegerValue,
    right: SupportsIndex | CanonicalInteger | IntegerValue,
) -> ExtendedGcdResult:
    """Return a gcd and exact Bezout coefficients for two integers."""

    from sympy import gcdex

    left_value = _as_python_integer(left)
    right_value = _as_python_integer(right)
    x, y, divisor = gcdex(left_value, right_value)
    return ExtendedGcdResult(
        gcd=format_canonical_integer(int(divisor)),
        left_coefficient=format_canonical_integer(int(x)),
        right_coefficient=format_canonical_integer(int(y)),
    )


def prime_valuation(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
    prime: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the exponent of a prime in one nonzero integer."""

    from sympy import isprime, multiplicity

    integer = _as_python_integer(value)
    prime_value = _as_python_integer(prime)
    if integer == 0:
        raise OperationDomainValidationError(
            location=("value",),
            code="number_theory.valuation_requires_nonzero_value",
            message="valuation requires nonzero value",
        )
    if prime_value < 2 or not isprime(prime_value):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.valuation_requires_a_prime_absolute_base_2",
            message="valuation requires a prime absolute base >= 2",
        )
    return IntegerValue(
        value=format_canonical_integer(multiplicity(prime_value, abs(integer)))
    )


def _positive_integer(value: SupportsIndex | CanonicalInteger | IntegerValue) -> int:
    integer = _as_python_integer(value)
    if integer <= 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.positive_integer_required",
            message="value must be positive",
        )
    return integer


def divisor_count(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the number of positive divisors of a positive integer."""

    from sympy import divisor_count as sympy_divisor_count

    return IntegerValue(
        value=format_canonical_integer(
            int(sympy_divisor_count(_positive_integer(value)))
        )
    )


def divisor_sum(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the sum of the positive divisors of a positive integer."""

    from sympy import divisor_sigma

    return IntegerValue(
        value=format_canonical_integer(int(divisor_sigma(_positive_integer(value))))
    )


def aliquot_sum(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the sum of the proper positive divisors of a positive integer."""

    integer = _positive_integer(value)
    from sympy import divisor_sigma

    return IntegerValue(
        value=format_canonical_integer(int(divisor_sigma(integer)) - integer)
    )


def are_coprime(
    left: SupportsIndex | CanonicalInteger | IntegerValue,
    right: SupportsIndex | CanonicalInteger | IntegerValue,
) -> BooleanResult:
    """Return whether two integers are coprime."""

    return BooleanResult(
        holds=gcd(_as_python_integer(left), _as_python_integer(right)) == 1
    )


def divides(
    divisor: SupportsIndex | CanonicalInteger | IntegerValue,
    dividend: SupportsIndex | CanonicalInteger | IntegerValue,
) -> BooleanResult:
    """Return whether one nonzero integer divides another."""

    divisor_value = _as_python_integer(divisor)
    if divisor_value == 0:
        raise OperationDomainValidationError(
            location=("divisor",),
            code="number_theory.divisor_must_be_nonzero",
            message="divisor must be nonzero",
        )
    return BooleanResult(holds=_as_python_integer(dividend) % divisor_value == 0)


def is_even(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether an integer is even."""

    return BooleanResult(holds=_as_python_integer(value) % 2 == 0)


def is_odd(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether an integer is odd."""

    return BooleanResult(holds=_as_python_integer(value) % 2 != 0)


def is_square(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether a nonnegative integer is a square."""

    integer = _as_python_integer(value)
    if integer < 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.nonnegative_integer_required",
            message="value must be nonnegative",
        )
    return BooleanResult(holds=isqrt(integer) ** 2 == integer)


def _aliquot_relation(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> tuple[int, int]:
    from sympy import divisor_sigma

    integer = _as_python_integer(value)
    if integer < 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.nonnegative_integer_required",
            message="value must be nonnegative",
        )
    return integer, int(divisor_sigma(integer)) - integer


def is_perfect(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether a positive integer equals its aliquot sum."""

    integer, aliquot = _aliquot_relation(value)
    return BooleanResult(holds=bool(integer and aliquot == integer))


def is_abundant(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> BooleanResult:
    """Return whether a positive integer is smaller than its aliquot sum."""

    integer, aliquot = _aliquot_relation(value)
    return BooleanResult(holds=bool(integer and aliquot > integer))


def is_deficient(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> BooleanResult:
    """Return whether a positive integer is larger than its aliquot sum."""

    integer, aliquot = _aliquot_relation(value)
    return BooleanResult(holds=bool(integer and aliquot < integer))


def absolute_value(value: SupportsIndex | IntegerValue) -> IntegerValue:
    """Return the canonical shared integer value of the exact absolute value."""

    return IntegerValue(value=format_canonical_integer(abs(_as_python_integer(value))))


def sign(value: SupportsIndex | IntegerValue) -> int:
    """Return -1, 0, or 1 according to the sign of an integer."""

    integer = _as_python_integer(value)
    return (integer > 0) - (integer < 0)


def decimal_digit_sum(value: IntegerValue) -> IntegerValue:
    """Return the sum of the decimal digits of an exact integer."""
    return IntegerValue(
        value=format_canonical_integer(
            sum(int(digit) for digit in value.value.lstrip("-"))
        )
    )


def decimal_digit_count(value: IntegerValue) -> IntegerValue:
    """Return the number of decimal digits in an exact integer's magnitude."""
    return IntegerValue(value=format_canonical_integer(len(value.value.lstrip("-"))))


def base_digits(value: IntegerValue, base: int) -> tuple[int, int, tuple[str, ...]]:
    """Return sign, base, and canonical positional digits for an integer."""
    if not 2 <= base <= _MAX_BASE:
        raise OperationDomainValidationError(
            location=("base",),
            code="arithmetic.base_out_of_range",
            message="base must be between 2 and 10000",
        )
    magnitude = value.value.lstrip("-")
    maximum_value = format_canonical_integer(base**MAX_BASE_DIGITS)
    if len(magnitude) > len(maximum_value) or (
        len(magnitude) == len(maximum_value) and magnitude >= maximum_value
    ):
        raise OperationDomainValidationError(
            location=("value",),
            code="arithmetic.base_expansion_exceeds_bound",
            message=f"base expansion exceeds the {MAX_BASE_DIGITS}-digit result bound",
        )
    from sympy.ntheory import digits as sympy_digits

    integer = parse_canonical_integer(value.value)
    signed_base, *expanded = sympy_digits(integer, base)
    sign_value = 0 if integer == 0 else (-1 if signed_base < 0 else 1)
    return sign_value, abs(signed_base), tuple(str(digit) for digit in expanded)


def nth_root(value: IntegerValue, degree: int) -> tuple[IntegerValue, bool]:
    """Return the exact floor nth root and whether the root is integral."""
    if not 1 <= degree <= _MAX_NTH_ROOT_DEGREE:
        raise OperationDomainValidationError(
            location=("degree",),
            code="arithmetic.root_degree_out_of_range",
            message="degree must be between 1 and 100000",
        )
    from sympy import integer_nthroot

    integer = parse_canonical_integer(value.value)
    if integer < 0 and degree % 2 == 0:
        raise OperationDomainValidationError(
            location=("value", "degree"),
            code="arithmetic.even_root_of_negative",
            message="even root of a negative integer is not integral-real",
        )
    root, exact = integer_nthroot(abs(integer), degree)
    if integer < 0 and not exact:
        root += 1
    return IntegerValue(
        value=format_canonical_integer(-root if integer < 0 else root)
    ), exact


def _as_rational(value: Fraction | int | IntegerValue) -> Fraction:
    """Return one admitted rational input as its exact Python fraction."""

    if isinstance(value, IntegerValue):
        return Fraction(parse_canonical_integer(value.value))
    return Fraction(value)


def reciprocal(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact reciprocal, rejecting zero."""

    rational = _as_rational(value)
    if not rational:
        raise OperationDomainValidationError(
            location=("value",),
            code="arithmetic.reciprocal_requires_nonzero",
            message="reciprocal requires a nonzero rational",
        )
    return 1 / rational


def sum_rationals(
    left: Fraction | int | IntegerValue, right: Fraction | int | IntegerValue
) -> Fraction:
    """Add two exact rational values."""

    return _as_rational(left) + _as_rational(right)


def negate_rational(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact additive inverse of a rational value."""

    return -_as_rational(value)


def rational_absolute_value(value: Fraction | int | IntegerValue) -> Fraction:
    """Return the exact absolute value of a rational value."""

    return abs(_as_rational(value))


def difference_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Subtract two exact rational values."""

    return _as_rational(left) - _as_rational(right)


def product_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Multiply two exact rational values."""

    return _as_rational(left) * _as_rational(right)


def minimum_rational(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Return the lesser of two exact rational values."""

    return min(_as_rational(left), _as_rational(right))


def maximum_rational(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> Fraction:
    """Return the greater of two exact rational values."""

    return max(_as_rational(left), _as_rational(right))


def floor_rational(value: Fraction | int | IntegerValue) -> int:
    """Return the greatest integer not exceeding an exact rational."""

    return floor(_as_rational(value))


def ceiling_rational(value: Fraction | int | IntegerValue) -> int:
    """Return the least integer not below an exact rational."""

    return ceil(_as_rational(value))


def continued_fraction(
    value: Fraction | int | IntegerValue,
    *,
    max_terms: int | None = None,
) -> tuple[int, ...]:
    """Return the canonical finite simple continued fraction of a rational."""

    return _continued_fraction_terms(_as_rational(value), max_terms=max_terms)


def _continued_fraction_terms(
    rational: Fraction,
    *,
    max_terms: int | None = None,
) -> tuple[int, ...]:
    """Expand one rational, stopping before a bounded result would overflow."""

    numerator = rational.numerator
    denominator = rational.denominator
    terms: list[int] = []
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        if max_terms is not None and len(terms) == max_terms:
            raise OperationDomainValidationError(
                location=("value",),
                code="arithmetic.continued_fraction_terms_exceed_limit",
                message=(
                    f"continued fraction exceeds the {max_terms}-term result bound"
                ),
            )
        terms.append(quotient)
        numerator, denominator = denominator, remainder
    return tuple(terms)


def equal_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> bool:
    """Decide exact rational equality."""

    return _as_rational(left) == _as_rational(right)


def less_than_rationals(
    left: Fraction | int | IntegerValue,
    right: Fraction | int | IntegerValue,
) -> bool:
    """Decide strict order of two exact rational values."""

    return _as_rational(left) < _as_rational(right)


def integerize_rational_vector(
    values: Iterable[Fraction | int | IntegerValue],
) -> tuple[int, ...]:
    """Scale exact rationals to integer coordinates with a shared denominator."""

    rationals = tuple(_as_rational(value) for value in values)
    common_denominator = lcm(*(value.denominator for value in rationals))
    return tuple(
        value.numerator * (common_denominator // value.denominator)
        for value in rationals
    )


def primitive_integer_vector(
    values: Iterable[Fraction | int | IntegerValue],
) -> tuple[int, ...]:
    """Normalize a nonzero rational vector to primitive, positive-leading integers."""

    integers = integerize_rational_vector(values)
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if not divisor:
        raise OperationDomainValidationError(
            location=("values",),
            code="arithmetic.primitive_vector_requires_nonzero",
            message="a primitive integer vector must be nonzero",
        )
    primitive = tuple(value // divisor for value in integers)
    if next(value for value in primitive if value) < 0:
        return tuple(-value for value in primitive)
    return primitive


def quotient(
    left: Fraction | int | IntegerValue, right: Fraction | int | IntegerValue
) -> Fraction:
    """Divide two exact rational values."""

    divisor = _as_rational(right)
    if not divisor:
        raise OperationDomainValidationError(
            location=("right",),
            code="arithmetic.division_requires_nonzero_divisor",
            message="quotient requires a nonzero divisor",
        )
    return _as_rational(left) / divisor
