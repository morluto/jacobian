"""Value-based native operations on canonical prime-affine tuple values."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.prime_affine_forms._models import (
    FinitePrimeTupleFactorProduct,
    PrimeAffineIntervalCountRequest,
    PrimeAffineIntervalEnumerateRequest,
    PrimeAffineTranslationRequest,
    PrimeAffineTranslationResult,
    PrimePatternIntervalCountResult,
    PrimePatternIntervalEnumerateResult,
    PrimeTupleAdmissibilityRequest,
    PrimeTupleAdmissibilityResult,
    PrimeTupleIntervalResidueProfileRequest,
    PrimeTupleIntervalResidueProfileResult,
    PrimeTupleLocalFactorRequest,
    PrimeTupleLocalFactorResult,
    PrimeTupleLocalFactorsRequest,
    PrimeTupleResidueWheel,
    PrimeTupleResidueWheelEnumeration,
    PrimeTupleResidueWheelEnumerationRequest,
    PrimeTupleResidueWheelRequest,
    PrimeTupleWheelMembershipRequest,
    PrimeTupleWheelMembershipResult,
)
from jacobian.math.prime_affine_forms._operations import (
    compute_interval_count,
    compute_interval_enumerate,
    compute_interval_residue_profile,
    compute_local_admissibility,
    compute_local_factor,
    compute_local_factors,
    compute_residue_wheel,
    compute_residue_wheel_enumeration,
    compute_translation,
    compute_wheel_membership,
)
from jacobian.math.prime_affine_forms.values import PrimeAffineTuple


def local_factor(source: PrimeAffineTuple, prime: int) -> PrimeTupleLocalFactorResult:
    """Return the complete modulo-``prime`` residue partition and local factor."""

    return compute_local_factor(
        PrimeTupleLocalFactorRequest(source=source, prime=prime)
    )


def local_factors(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> FinitePrimeTupleFactorProduct:
    """Return exact compact local factors over one finite prime set."""

    return compute_local_factors(
        PrimeTupleLocalFactorsRequest(source=source, primes=primes)
    )


def local_admissibility(source: PrimeAffineTuple) -> PrimeTupleAdmissibilityResult:
    """Decide local admissibility by checking exactly every prime at most k."""

    return compute_local_admissibility(PrimeTupleAdmissibilityRequest(source=source))


def residue_wheel(
    source: PrimeAffineTuple, primes: tuple[int, ...]
) -> PrimeTupleResidueWheel:
    """Construct the compact source-bound CRT wheel of one finite prime set."""

    return compute_residue_wheel(
        PrimeTupleResidueWheelRequest(source=source, primes=primes)
    )


def enumerate_residue_wheel(
    wheel: PrimeTupleResidueWheel,
) -> PrimeTupleResidueWheelEnumeration:
    """Materialize every permitted CRT residue of a supplied compact wheel."""

    return compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=wheel)
    )


def wheel_membership(
    wheel: PrimeTupleResidueWheel, value: int
) -> PrimeTupleWheelMembershipResult:
    """Reduce one exact integer through a source-bound residue wheel."""

    return compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(
            wheel=wheel, value=format_canonical_integer(value)
        )
    )


def interval_count(
    source: PrimeAffineTuple, lower: int, upper: int
) -> PrimePatternIntervalCountResult:
    """Count every n in [lower, upper] whose affine values are all positive primes."""

    return compute_interval_count(
        PrimeAffineIntervalCountRequest(
            source=source,
            lower=format_canonical_integer(lower),
            upper=format_canonical_integer(upper),
        )
    )


def interval_enumerate(
    source: PrimeAffineTuple, lower: int, upper: int
) -> PrimePatternIntervalEnumerateResult:
    """Enumerate every n in [lower, upper] whose affine values are all positive primes."""

    return compute_interval_enumerate(
        PrimeAffineIntervalEnumerateRequest(
            source=source,
            lower=format_canonical_integer(lower),
            upper=format_canonical_integer(upper),
        )
    )


def interval_residue_profile(
    wheel: PrimeTupleResidueWheel, lower: int, upper: int
) -> PrimeTupleIntervalResidueProfileResult:
    """Enumerate wheel-permitted integers on one bounded closed interval."""

    return compute_interval_residue_profile(
        PrimeTupleIntervalResidueProfileRequest(
            wheel=wheel,
            lower=format_canonical_integer(lower),
            upper=format_canonical_integer(upper),
        )
    )


def translate_tuple(
    source: PrimeAffineTuple, shift: int
) -> PrimeAffineTranslationResult:
    """Apply the substitution n -> n+shift to every labelled form."""

    return compute_translation(
        PrimeAffineTranslationRequest(
            source=source, shift=format_canonical_integer(shift)
        )
    )


__all__ = [
    "enumerate_residue_wheel",
    "interval_count",
    "interval_enumerate",
    "interval_residue_profile",
    "local_admissibility",
    "local_factor",
    "local_factors",
    "residue_wheel",
    "translate_tuple",
    "wheel_membership",
]
