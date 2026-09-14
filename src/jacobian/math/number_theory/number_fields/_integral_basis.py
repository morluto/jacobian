"""Private SymPy kernel for canonical simple-field presentations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS,
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
_ROUND_TWO_COFACTOR_LIMIT = 10**_ROUND_TWO_COFACTOR_DIGIT_BOUND


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


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value))) if value else 1


def _monicization_digit_growth(leading: int) -> int:
    """Return the digits one multiplication by ``leading`` can add.

    Monicizing ``f`` to ``A^(n-1) f(x/A)`` scales the coefficient at index
    ``i`` by ``A^(i-1)``, so each power of the leading coefficient adds at most
    ``ceil(log10 A)`` digits. A unit leading coefficient multiplies by one and
    adds nothing: counting its one digit as growth charges ``i - 1`` phantom
    digits to every monic field and rejects valid discriminants that already
    fit the declared carrier.
    """

    magnitude = abs(leading)
    if magnitude <= 1:
        return 0
    return _integer_digits(magnitude - 1)


def monicized_discriminant_digit_bound(
    field: SimpleNumberFieldPresentation,
) -> int:
    """Return a digit envelope for ``disc`` of ``A^(n-1) f(x/A)``."""

    coefficients = tuple(
        int(coefficient) for coefficient in field.coefficients_descending
    )
    leading = abs(coefficients[0])
    growth = _monicization_digit_growth(leading)
    widest = 1
    for index, coefficient in enumerate(coefficients[1:], start=1):
        coeff_digits = _integer_digits(abs(coefficient))
        widest = max(widest, coeff_digits + (index - 1) * growth)
    degree = field.degree
    return max(1, (2 * degree - 1) * widest + 4 * degree)


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


def _factorize_small_factors(value: int) -> tuple[int, dict[int, int]]:
    """Remove every prime factor up to the trial bound from ``value``.

    Returns the stripped cofactor together with the small-prime exponents so
    admission can later hand the complete factorization back to the backend.
    """

    cofactor = value
    factors: dict[int, int] = {}
    for prime in _trial_primes(_ROUND_TWO_TRIAL_PRIME_BOUND):
        if prime * prime > cofactor:
            break
        exponent = 0
        while cofactor % prime == 0:
            cofactor //= prime
            exponent += 1
        if exponent:
            factors[prime] = exponent
        if cofactor == 1:
            break
    return cofactor, factors


def _cofactor_prime_power(cofactor: int) -> tuple[int, int] | None:
    """Return the proved prime base and exponent of a stripped cofactor.

    ``cofactor`` must already be stripped of every prime factor up to the
    trial bound. factorint then terminates through its exact
    prime/perfect-power short-circuits without search exactly when the cofactor
    is prime or a prime power; return ``None`` otherwise.
    """

    from sympy.ntheory import isprime, perfect_power

    candidate: int | bool = cofactor
    exponent = 1
    while isinstance(candidate, int):
        if candidate == 1:
            return None
        if candidate >= _ROUND_TWO_COFACTOR_LIMIT:
            return None
        power = perfect_power(candidate, factor=False)
        if power is False:
            return (candidate, exponent) if isprime(candidate) else None
        candidate, step = power
        exponent *= int(step)
    return None


def _seed_round_two_factor_cache(
    discriminant: int, factorization: dict[int, int]
) -> None:
    """Seed SymPy's factor cache so ``round_two`` reuses the admitted work.

    ``round_two`` factors the same discriminant again through ``factorint``.
    Reproducing that call's initial trial-division remainder and recording the
    already-proved prime there lets the backend skip the repeated primality
    test.  This is only a hint; correctness never depends on the cache.
    """

    from sympy import factor_cache
    from sympy.ntheory.factor_ import _factorint_small

    remaining, _ = _factorint_small({}, abs(discriminant), 2**15, 600)
    remaining = int(remaining)
    while remaining > 1:
        for prime in sorted(factorization):
            if remaining % prime == 0:
                factor_cache[remaining] = int(prime)
                remaining //= prime ** factorization[prime]
                break
        else:
            # An unrecognized remainder shape: leave the cache untouched and
            # let the backend factor it itself.
            return


def require_factorizable_discriminant(
    field: SimpleNumberFieldPresentation,
) -> int | None:
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

    The computed monic polynomial discriminant is returned for this request
    so the backend can reuse it; it is not retained in module-global state.
    """

    if field.degree < 1:
        return None
    estimated_digits = monicized_discriminant_digit_bound(field)
    if estimated_digits > MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="number_field.ring_of_integers_discriminant_output_bound",
            message=(
                "the monicized defining-polynomial discriminant exceeds the "
                f"{MAX_NUMBER_FIELD_DISCRIMINANT_DIGITS}-digit result envelope"
            ),
        )
    request_checkpoint("before number-field discriminant admission")
    import sympy

    monic_coefficients = _monic_zz_coefficients(field)
    polynomial = sympy.Poly.from_list(
        list(monic_coefficients),
        gens=sympy.Symbol("alpha"),
        domain=sympy.ZZ,
    )
    if polynomial.is_irreducible is not True:
        return None
    discriminant = int(polynomial.discriminant())
    if discriminant == 0:
        return None
    cofactor, small_factors = _factorize_small_factors(abs(discriminant))
    factorization = dict(small_factors)
    if cofactor != 1:
        prime_power = _cofactor_prime_power(cofactor)
        if prime_power is None:
            raise OperationResourceAdmissionError(
                location=("field",),
                code="number_field.ring_of_integers_discriminant_factorization_bound",
                message=(
                    "the defining-polynomial discriminant has a composite "
                    "cofactor beyond the bounded trial envelope, so round_two "
                    "factorization work is not soundly bounded"
                ),
            )
        base, exponent = prime_power
        factorization[base] = factorization.get(base, 0) + exponent
    _seed_round_two_factor_cache(discriminant, factorization)
    return discriminant


def _poly_with_admitted_round_two_work(
    polynomial: Any,
    *,
    admitted_discriminant: int | None,
    admitted_irreducible: bool,
) -> Any:
    """Supply request-scoped round_two facts without mutating Poly methods."""

    import sympy

    admitted_value = (
        None if admitted_discriminant is None else sympy.Integer(admitted_discriminant)
    )
    poly_type = type(polynomial)

    class _AdmittedRoundTwoPoly(poly_type):  # type: ignore[misc, valid-type]
        def discriminant(self, *args: object, **kwargs: object) -> Any:
            if admitted_value is None:
                return super().discriminant(*args, **kwargs)
            return admitted_value

        @property
        def is_irreducible(self) -> bool:
            if admitted_irreducible:
                return True
            return bool(super().is_irreducible)

    return _AdmittedRoundTwoPoly(
        polynomial.as_expr(),
        *polynomial.gens,
        domain=polynomial.domain,
    )


def recognized_integral_basis(
    field: SimpleNumberFieldPresentation,
    admitted_polynomial_discriminant: int | None = None,
    *,
    admitted_irreducible: bool | None = None,
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
    if admitted_irreducible is False:
        return None
    if admitted_irreducible is not True and polynomial.is_irreducible is not True:
        return None
    if admitted_polynomial_discriminant is not None or admitted_irreducible is True:
        polynomial = _poly_with_admitted_round_two_work(
            polynomial,
            admitted_discriminant=admitted_polynomial_discriminant,
            admitted_irreducible=admitted_irreducible is True,
        )
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
    # The coordinate denominators divide the index of Z[alpha] in the maximal
    # order, which is the determinant of the recognized HNF basis relative to
    # the power basis. Admit that growth before expanding any coordinate.
    index = abs(int(ring.matrix.det()))
    if _integer_digits(index) > MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="number_field.integral_basis_coordinate_bound",
            message=(
                "the index of the presented power order exceeds the admitted "
                f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit element envelope"
            ),
        )
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
            try:
                require_bounded_rational(
                    coefficient,
                    max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
                    label="integral basis",
                )
            except ValueError as exc:
                # The integral-basis coordinate denominators divide the index
                # of Z[alpha], which can exceed the element carrier even when
                # the defining discriminant fits its published envelope. Map
                # that case to a typed admission error so the worker reports a
                # rejection rather than crashing the parent.
                raise OperationResourceAdmissionError(
                    location=("field",),
                    code="number_field.integral_basis_coordinate_bound",
                    message=(
                        "an integral-basis coordinate exceeds the admitted "
                        f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit element "
                        "envelope"
                    ),
                ) from exc
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
    "monicized_discriminant_digit_bound",
    "recognized_integral_basis",
    "require_factorizable_discriminant",
]
