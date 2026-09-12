"""Private SymPy kernel for canonical simple-field presentations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldPresentation,
)

# Trial-division ceiling for discriminant admission. SymPy's factorint removes
# small factors with trial primes up to 2**15 before its primality and
# perfect-power short-circuits; stripping every prime factor up to 10**5 first
# guarantees the remaining cofactor is exactly what those short-circuits
# decide, so round_two never reaches Pollard-Rho/ECM search.
_ROUND_TWO_TRIAL_PRIME_BOUND = 100_000

# Largest cofactor (in decimal digits) admitted to the exact primality and
# perfect-power checks. Both checks run in time polynomial in the digit count,
# so this cap keeps admission itself soundly bounded; larger cofactors cannot
# be proven factorable here and are rejected.
_ROUND_TWO_COFACTOR_DIGIT_BOUND = 4096


def _monic_zz_coefficients(
    field: SimpleNumberFieldPresentation,
) -> tuple[int, ...]:
    """Return the monic integral polynomial for ``beta = L*alpha``.

    This is the exact polynomial whose discriminant SymPy's round_two
    factors, so admission must decide on these same coefficients.
    """

    coefficients = tuple(
        int(coefficient) for coefficient in field.coefficients_descending
    )
    leading = coefficients[0]
    # beta = leading * alpha has the monic integral polynomial
    # leading^(n-1) f(beta / leading). This preserves QQ(alpha) while
    # allowing the canonical presentation itself to remain nonmonic.
    return (
        1,
        *(
            coefficient * leading ** (index - 1)
            for index, coefficient in enumerate(coefficients[1:], start=1)
        ),
    )


def _trial_primes(limit: int) -> tuple[int, ...]:
    """Return every prime up to ``limit`` by an exact sieve."""

    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for factor in range(2, int(limit**0.5) + 1):
        if sieve[factor]:
            sieve[factor * factor :: factor] = b"\x00" * (
                (limit - factor * factor) // factor + 1
            )
    return tuple(index for index, prime in enumerate(sieve) if prime)


def _strip_small_factors(value: int) -> int:
    """Remove every prime factor up to the trial bound from ``value``."""

    cofactor = value
    for prime in _trial_primes(_ROUND_TWO_TRIAL_PRIME_BOUND):
        if prime * prime > cofactor:
            break
        while cofactor % prime == 0:
            cofactor //= prime
        if cofactor == 1:
            break
    return cofactor


def _cofactor_is_factorizable(cofactor: int) -> bool:
    """Decide whether SymPy's factorint short-circuits on ``cofactor``.

    ``cofactor`` must already be stripped of every prime factor up to the
    trial bound. factorint then terminates through its exact
    prime/perfect-power checks without search exactly when the cofactor is
    one, prime, or a prime power.
    """

    from sympy.ntheory import isprime, perfect_power

    candidate: int | bool = cofactor
    while isinstance(candidate, int):
        if candidate == 1:
            return True
        if len(str(candidate)) > _ROUND_TWO_COFACTOR_DIGIT_BOUND:
            return False
        power = perfect_power(candidate)
        if power is False:
            return isprime(candidate)
        candidate, _ = power


def require_factorizable_discriminant(
    field: SimpleNumberFieldPresentation,
) -> None:
    """Admit the round_two discriminant work before any backend expansion.

    SymPy's round_two completely factors the discriminant of the monicized
    integral polynomial. A discriminant such as ``4N`` for ``x^2 - N`` with
    ``N`` a product of two large primes fits the coefficient carrier but
    makes that factorization infeasible, so the worker would time out instead
    of establishing the advertised exact basis. This check proves, using only
    exact polynomial-time work, that factorint terminates through its
    prime/perfect-power short-circuits, or rejects the field with a typed
    domain error. Reducible presentations have a zero discriminant and flow
    to the existing irreducibility error downstream.
    """

    if field.degree < 1:
        return
    request_checkpoint("before number-field discriminant admission")
    import sympy

    polynomial = sympy.Poly.from_list(
        list(_monic_zz_coefficients(field)),
        gens=sympy.Symbol("alpha"),
        domain=sympy.ZZ,
    )
    discriminant = int(polynomial.discriminant())
    if discriminant == 0:
        return
    cofactor = _strip_small_factors(abs(discriminant))
    if _cofactor_is_factorizable(cofactor):
        return
    raise OperationDomainValidationError(
        location=("field",),
        code="number_field.ring_of_integers_discriminant_factorization_bound",
        message=(
            "the defining-polynomial discriminant has a composite cofactor "
            "beyond the bounded trial envelope, so round_two factorization "
            "work is not soundly bounded"
        ),
    )


def recognized_integral_basis(
    field: SimpleNumberFieldPresentation,
) -> tuple[Any, Any, Any, int] | None:
    """Recognize the presentation and compute its integral basis once."""

    import sympy
    from sympy.polys.numberfields import round_two

    alpha = sympy.Symbol("alpha")
    monic_coefficients = _monic_zz_coefficients(field)
    leading = int(field.coefficients_descending[0])
    polynomial = sympy.Poly.from_list(
        list(monic_coefficients),
        gens=alpha,
        domain=sympy.ZZ,
    )
    if polynomial.is_irreducible is not True:
        return None
    ring, field_discriminant = cast(tuple[Any, Any], round_two(polynomial))
    return ring, field_discriminant, alpha, leading


def integral_basis_coordinates(
    field: SimpleNumberFieldPresentation,
    recognized: tuple[Any, Any, Any, int] | None = None,
) -> tuple[tuple[CanonicalRational, ...], ...] | None:
    """Return the recognized basis on the presented generator's power basis."""

    if recognized is None:
        recognized = recognized_integral_basis(field)
    if recognized is None:
        return None
    ring, _field_discriminant, alpha, leading = recognized
    basis: list[tuple[CanonicalRational, ...]] = []
    for element in ring.basis_element_pullbacks():
        expression = element.as_expr().subs(alpha, leading * alpha).expand()
        polynomial = _as_poly_in_alpha(expression, alpha)
        coefficients = polynomial.all_coeffs()[::-1]
        padded = [_rational(coefficient) for coefficient in coefficients]
        padded.extend(
            CanonicalRational(num=0, den=1) for _ in range(field.degree - len(padded))
        )
        if len(padded) != field.degree:
            raise ValueError(
                "an integral basis vector must span the complete power basis"
            )
        for coefficient in padded:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
                label="integral basis",
            )
        basis.append(tuple(padded))
    return tuple(basis)


def _rational(value: Any) -> CanonicalRational:
    fraction = Fraction(value)
    return CanonicalRational(num=fraction.numerator, den=fraction.denominator)


def _as_poly_in_alpha(expression: Any, alpha: Any) -> Any:
    import sympy

    return sympy.Poly(expression, alpha)


__all__ = [
    "integral_basis_coordinates",
    "recognized_integral_basis",
    "require_factorizable_discriminant",
]
