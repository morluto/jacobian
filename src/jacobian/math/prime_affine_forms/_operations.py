"""Domain operations for exact prime-affine local arithmetic."""

from __future__ import annotations

from math import prod

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.prime_affine_forms._kernel import (
    interval_match_summary,
    interval_matches,
    translated_tuple,
    wheel_modulus,
    wheel_rows,
)
from jacobian.math.prime_affine_forms._local_factors import local_summary
from jacobian.math.prime_affine_forms._models import (
    PrimeAffineIntervalCountRequest,
    PrimeAffineIntervalEnumerateRequest,
    PrimeAffineTranslationRequest,
    PrimeAffineTranslationResult,
    PrimePatternIntervalCountResult,
    PrimePatternIntervalEnumerateResult,
    PrimePatternMatch,
)
from jacobian.math.prime_affine_forms._residue_wheel import (
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
    PrimeTupleWheelMembershipResult,
    PrimeTupleWheelResidueRow,
)


def compute_residue_wheel(
    request: PrimeTupleResidueWheelRequest,
) -> PrimeTupleResidueWheel:
    local_rows = tuple(local_summary(request.source, prime) for prime in request.primes)
    return PrimeTupleResidueWheel(
        source=request.source,
        primes=request.primes,
        local_rows=local_rows,
        modulus=format_canonical_integer(wheel_modulus(request.primes)),
        valid_count=format_canonical_integer(
            prod((row.valid_count for row in local_rows), start=1)
        ),
    )


def compute_residue_wheel_enumeration(
    request: PrimeTupleResidueWheelEnumerationRequest,
) -> PrimeTupleResidueWheelEnumeration:
    rows = wheel_rows(request.wheel.source, request.wheel.primes)
    return PrimeTupleResidueWheelEnumeration(
        wheel=request.wheel,
        residues=tuple(
            PrimeTupleWheelResidueRow(
                residue=format_canonical_integer(residue),
                components=components,
            )
            for residue, components in rows
        ),
    )


def compute_wheel_membership(
    request: PrimeTupleWheelMembershipRequest,
) -> PrimeTupleWheelMembershipResult:
    value = parse_canonical_integer(request.value)
    modulus = parse_canonical_integer(request.wheel.modulus)
    residue = value % modulus
    components = tuple(value % prime for prime in request.wheel.primes)
    first_prime: int | None = None
    form_ids: tuple[str, ...] = ()
    for summary, component in zip(request.wheel.local_rows, components, strict=True):
        bad = {row.residue: row.form_ids for row in summary.bad_residues}
        if component in bad:
            first_prime = summary.prime
            form_ids = bad[component]
            break
    return PrimeTupleWheelMembershipResult(
        wheel=request.wheel,
        value=request.value,
        canonical_residue=format_canonical_integer(residue),
        components=components,
        is_permitted=first_prime is None,
        first_excluded_prime=first_prime,
        vanishing_form_ids=form_ids,
    )


def compute_interval_count(
    request: PrimeAffineIntervalCountRequest,
) -> PrimePatternIntervalCountResult:
    lower = parse_canonical_integer(request.lower)
    upper = parse_canonical_integer(request.upper)
    interval_size = upper - lower + 1
    count, first, last = interval_match_summary(request.source, lower, upper)
    return PrimePatternIntervalCountResult(
        source=request.source,
        lower=request.lower,
        upper=request.upper,
        interval_size=interval_size,
        affine_values_examined=interval_size * request.source.form_count,
        match_count=count,
        first_match=None if first is None else format_canonical_integer(first),
        last_match=None if last is None else format_canonical_integer(last),
    )


def compute_interval_enumerate(
    request: PrimeAffineIntervalEnumerateRequest,
) -> PrimePatternIntervalEnumerateResult:
    lower = parse_canonical_integer(request.lower)
    upper = parse_canonical_integer(request.upper)
    interval_size = upper - lower + 1
    matches = interval_matches(request.source, lower, upper)
    return PrimePatternIntervalEnumerateResult(
        source=request.source,
        lower=request.lower,
        upper=request.upper,
        interval_size=interval_size,
        affine_values_examined=interval_size * request.source.form_count,
        matches=tuple(
            PrimePatternMatch(
                parameter=format_canonical_integer(parameter),
                prime_values=tuple(format_canonical_integer(value) for value in values),
            )
            for parameter, values in matches
        ),
    )


def compute_interval_residue_profile(
    request: PrimeTupleIntervalResidueProfileRequest,
) -> PrimeTupleIntervalResidueProfileResult:
    lower = parse_canonical_integer(request.lower)
    upper = parse_canonical_integer(request.upper)
    bad_by_prime = tuple(
        {row.residue for row in summary.bad_residues}
        for summary in request.wheel.local_rows
    )
    survivors = tuple(
        value
        for value in range(lower, upper + 1)
        if all(
            value % prime not in bad
            for prime, bad in zip(request.wheel.primes, bad_by_prime, strict=True)
        )
    )
    return PrimeTupleIntervalResidueProfileResult(
        wheel=request.wheel,
        lower=request.lower,
        upper=request.upper,
        interval_size=upper - lower + 1,
        survivors=tuple(format_canonical_integer(value) for value in survivors),
    )


def compute_translation(
    request: PrimeAffineTranslationRequest,
) -> PrimeAffineTranslationResult:
    return PrimeAffineTranslationResult(
        source=request.source,
        shift=request.shift,
        translated=translated_tuple(
            request.source, parse_canonical_integer(request.shift)
        ),
    )


__all__ = [
    "compute_interval_count",
    "compute_interval_enumerate",
    "compute_interval_residue_profile",
    "compute_residue_wheel",
    "compute_residue_wheel_enumeration",
    "compute_translation",
    "compute_wheel_membership",
]
