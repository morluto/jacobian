"""Exact finite kernels for prime-affine local arithmetic."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import prod
from typing import TYPE_CHECKING

from sympy import isprime, primerange
from sympy.ntheory.modular import crt1, crt2

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jacobian.math.number_theory.prime_affine_forms.values import PrimeAffineTuple

MAX_DETERMINISTIC_PRIME_INPUT = 2**64 - 1


def local_bad_residues(
    source: PrimeAffineTuple, prime: int
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return nonempty residue/form incidence rows in canonical residue order."""

    by_residue: dict[int, list[str]] = {}
    for form in source.forms:
        coefficient = form.coefficient
        constant = form.constant
        if coefficient % prime == 0:
            continue
        residue = (-constant * pow(coefficient, -1, prime)) % prime
        by_residue.setdefault(residue, []).append(form.form_id)
    return tuple(
        (residue, tuple(sorted(form_ids)))
        for residue, form_ids in sorted(by_residue.items())
    )


def local_counts(source: PrimeAffineTuple, prime: int) -> tuple[int, int]:
    bad_count = len(local_bad_residues(source, prime))
    return bad_count, prime - bad_count


def local_factor_from_bad_count(
    form_count: int, prime: int, bad_count: int
) -> Fraction:
    if bad_count == prime:
        return Fraction(0, 1)
    return Fraction(
        (prime - bad_count) * pow(prime, form_count - 1),
        pow(prime - 1, form_count),
    )


def valid_residues(source: PrimeAffineTuple, prime: int) -> tuple[int, ...]:
    bad = {residue for residue, _ in local_bad_residues(source, prime)}
    return tuple(residue for residue in range(prime) if residue not in bad)


def primes_through(bound: int) -> tuple[int, ...]:
    """Enumerate exactly the primes through the admitted finite cutoff."""

    return tuple(int(prime) for prime in primerange(2, bound + 1))


def wheel_modulus(primes: tuple[int, ...]) -> int:
    return prod(primes, start=1)


def iter_wheel_rows(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Yield checked CRT rows after distinct-prime and output preflight.

    Reuse the maintained backend's setup for fixed moduli and stream the
    cartesian component order. The consumer sorts its final canonical rows
    by residue; the kernel retains no complete intermediate row collection.
    Never scan the possibly enormous product modulus to obtain sorted rows.
    """

    if not primes:
        yield 0, ()
        return
    local_sets: list[tuple[int, ...]] = []
    for prime in primes:
        local = valid_residues(source, prime)
        if not local:
            return
        local_sets.append(local)
    if len(primes) == 1:
        for residue in local_sets[0]:
            yield residue, (residue,)
        return

    expected_modulus = wheel_modulus(primes)
    modulus, factors, inverses = crt1(primes)
    if int(modulus) != expected_modulus:
        raise RuntimeError("CRT setup failed its defining modulus invariant")
    for components in product(*local_sets):
        combined = crt2(primes, components, modulus, factors, inverses, symmetric=False)
        residue, returned_modulus = int(combined[0]), int(combined[1])
        # Replay against the original cartesian components, not values
        # reconstructed from the backend residue itself.
        if (
            returned_modulus != expected_modulus
            or not 0 <= residue < expected_modulus
            or any(
                residue % prime != component
                for prime, component in zip(primes, components, strict=True)
            )
        ):
            raise RuntimeError("CRT result failed its defining congruence invariant")
        yield residue, components


def iter_interval_values(
    source: PrimeAffineTuple, lower: int, upper: int
) -> Iterator[tuple[int, tuple[int, ...]]]:
    for parameter in range(lower, upper + 1):
        yield parameter, tuple(form.evaluate(parameter) for form in source.forms)


def is_positive_prime(value: int) -> bool:
    """Return exact ordinary-prime status in the admitted deterministic range."""

    return value > 1 and bool(isprime(value))


def interval_matches(
    source: PrimeAffineTuple, lower: int, upper: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (parameter, values)
        for parameter, values in iter_interval_values(source, lower, upper)
        if all(is_positive_prime(value) for value in values)
    )


def interval_match_summary(
    source: PrimeAffineTuple, lower: int, upper: int
) -> tuple[int, int | None, int | None]:
    count = 0
    first: int | None = None
    last: int | None = None
    for parameter, values in iter_interval_values(source, lower, upper):
        if all(is_positive_prime(value) for value in values):
            count += 1
            if first is None:
                first = parameter
            last = parameter
    return count, first, last


def translated_tuple(source: PrimeAffineTuple, shift: int) -> PrimeAffineTuple:
    from jacobian.math.number_theory.prime_affine_forms.values import (
        PrimeAffineTuple,
        PrimitiveIntegerAffineForm,
    )

    return PrimeAffineTuple(
        forms=tuple(
            PrimitiveIntegerAffineForm(
                form_id=form.form_id,
                coefficient=form.coefficient,
                constant=form.constant + form.coefficient * shift,
            )
            for form in source.forms
        )
    )


__all__ = [
    "MAX_DETERMINISTIC_PRIME_INPUT",
    "interval_match_summary",
    "interval_matches",
    "is_positive_prime",
    "iter_interval_values",
    "iter_wheel_rows",
    "local_bad_residues",
    "local_counts",
    "local_factor_from_bad_count",
    "primes_through",
    "translated_tuple",
    "valid_residues",
    "wheel_modulus",
]
