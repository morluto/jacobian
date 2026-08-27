"""Exact finite congruence-union operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory._periodic_kernel import (
    common_period,
    materialize_periodic_union,
    measure_periodic_union,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionMeasureResult,
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
    PeriodicCongruenceUnionRequest,
    PeriodicCongruenceUnionSource,
)


def _measure_request_from_source(
    source: PeriodicCongruenceUnionSource,
) -> PeriodicCongruenceUnionRequest:
    return PeriodicCongruenceUnionRequest.model_validate(source.model_dump(mode="json"))


def _measure_values(
    source: PeriodicCongruenceUnionSource,
    occupied_count: int,
) -> tuple[str, str, CanonicalRational]:
    period = common_period(source)
    return (
        format_canonical_integer(period),
        format_canonical_integer(occupied_count),
        CanonicalRational.from_fraction(Fraction(occupied_count, period)),
    )


def compute_periodic_congruence_union_measure(
    request: PeriodicCongruenceUnionRequest,
) -> PeriodicCongruenceUnionMeasureResult:
    """Compute the exact count and density of a finite congruence union."""

    source = request.normalized_source()
    period, occupied_count, density = _measure_values(
        source, measure_periodic_union(source)
    )
    return PeriodicCongruenceUnionMeasureResult._from_kernel(
        source=source,
        common_period=period,
        occupied_count=occupied_count,
        density=density,
    )


def compute_periodic_congruence_union_profile(
    request: PeriodicCongruenceUnionProfileRequest,
) -> PeriodicCongruenceUnionProfileResult:
    """Materialize the complete common-period residue profile."""

    source = request.normalized_source()
    residues = materialize_periodic_union(source)
    period, occupied_count, density = _measure_values(source, len(residues))
    return PeriodicCongruenceUnionProfileResult._from_profile_kernel(
        source=source,
        common_period=period,
        occupied_count=occupied_count,
        density=density,
        occupied_residues=tuple(
            format_canonical_integer(residue) for residue in residues
        ),
    )


def _verify_periodic_congruence_union_measure_result(
    result: PeriodicCongruenceUnionMeasureResult,
) -> bool:
    """Verify a deliberately supplied measure claim inside its admission envelope."""

    try:
        return (
            compute_periodic_congruence_union_measure(
                _measure_request_from_source(result.source)
            )
            == result
        )
    except ValueError:
        return False


def _verify_periodic_congruence_union_profile_result(
    result: PeriodicCongruenceUnionProfileResult,
) -> bool:
    """Verify a deliberately supplied materialized profile claim."""

    try:
        request = PeriodicCongruenceUnionProfileRequest.model_validate(
            result.source.model_dump(mode="json")
        )
        return compute_periodic_congruence_union_profile(request) == result
    except ValueError:
        return False


__all__ = [
    "compute_periodic_congruence_union_measure",
    "compute_periodic_congruence_union_profile",
]
