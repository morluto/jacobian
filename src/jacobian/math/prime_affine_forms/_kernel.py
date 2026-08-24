"""Exact finite kernels for prime-affine local arithmetic."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import prod
from typing import TYPE_CHECKING

from sympy import isprime, primerange
from sympy.ntheory.modular import crt

from jacobian.canonical import format_canonical_integer, parse_canonical_integer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jacobian.math.prime_affine_forms.values import PrimeAffineTuple

MAX_DETERMINISTIC_PRIME_INPUT = 2**64 - 1


def local_bad_residues(
    source: PrimeAffineTuple, prime: int
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return nonempty residue/form incidence rows in canonical residue order."""

    by_residue: dict[int, list[str]] = {}
    for form in source.forms:
        coefficient = parse_canonical_integer(form.coefficient)
        constant = parse_canonical_integer(form.constant)
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


def wheel_rows(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Enumerate CRT rows after distinct-prime and output preflight."""

    if not primes:
        return ((0, ()),)
    expected_modulus = wheel_modulus(primes)
    local_sets = tuple(valid_residues(source, prime) for prime in primes)
    if any(not residues for residues in local_sets):
        return ()
    rows: list[tuple[int, tuple[int, ...]]] = []
    for components in product(*local_sets):
        combined = crt(primes, components, symmetric=False, check=False)
        if combined is None:  # Pairwise-distinct primes make this unreachable.
            raise RuntimeError("CRT failed for pairwise-distinct prime moduli")
        residue, modulus = (int(combined[0]), int(combined[1]))
        if (
            modulus != expected_modulus
            or not 0 <= residue < expected_modulus
            or any(
                residue % prime != component
                for prime, component in zip(primes, components, strict=True)
            )
        ):
            raise RuntimeError("CRT result failed its defining congruence invariant")
        rows.append((residue, tuple(components)))
    return tuple(sorted(rows))


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
    from jacobian.math.prime_affine_forms.values import (
        PrimeAffineTuple,
        PrimitiveIntegerAffineForm,
    )

    return PrimeAffineTuple(
        forms=tuple(
            PrimitiveIntegerAffineForm(
                form_id=form.form_id,
                coefficient=form.coefficient,
                constant=format_canonical_integer(
                    parse_canonical_integer(form.constant)
                    + parse_canonical_integer(form.coefficient) * shift
                ),
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
    "local_bad_residues",
    "local_counts",
    "local_factor_from_bad_count",
    "primes_through",
    "translated_tuple",
    "valid_residues",
    "wheel_modulus",
    "wheel_rows",
]
