"""Certified finite quadratic-surd arithmetic on exact integer squares.

Every enclosure below is derived from ``isqrt`` of an exact integer, so the
bounds are exact rationals rather than floating-point approximations, and the
only irrational input is a nonsquare radicand.  Refinement is monotone because
a larger binary scale only subdivides the same integer-square bracket.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.number_theory.diophantine_approximation._surd_models import (
    MAX_SURD_MULTIPLIER_BITS,
    MAX_SURD_RADICAND,
    MAX_SURD_RANGE_INTERMEDIATE_BITS,
    MAX_SURD_RANGE_OUTPUT_BYTES,
    MAX_SURD_RANGE_WORK,
    MAX_SURD_SCALE_BITS,
    NearestIntegerDistanceRequest,
    NearestIntegerDistanceValue,
    RangeProfileRequest,
    RangeProfileResult,
    RangeProfileRow,
    RecordMinimaRequest,
    RecordMinimaResult,
    RecordMinimumValue,
    ScaledFloorRequest,
    ScaledFloorValue,
    SimultaneousProductRequest,
    SimultaneousProductResult,
    bound_enclosure,
    is_surd_radicand,
)

__all__ = [
    "nearest_integer_distance",
    "range_profile",
    "record_minima",
    "scaled_floor",
    "simultaneous_product",
]


def _require_request_multiplier(multiplier: int) -> None:
    if multiplier.bit_length() > MAX_SURD_MULTIPLIER_BITS:
        raise OperationResourceAdmissionError(
            location=("multiplier",),
            code="diophantine.multiplier_bit_bound",
            message="multiplier exceeds the 4096-bit exact-work bound",
        )
    if multiplier < 1:
        raise OperationDomainValidationError(
            location=("multiplier",),
            code="diophantine.multiplier_out_of_range",
            message="multiplier must be at least 1",
        )


def _require_surd_radicand(radicand: int, *, location: tuple[str, ...]) -> None:
    if not (2 <= radicand <= MAX_SURD_RADICAND) or not is_surd_radicand(radicand):
        raise OperationDomainValidationError(
            location=location,
            code="diophantine.surd_radicand_must_not_be_square",
            message="a quadratic irrational requires a nonsquare radicand",
        )


def _require_surd_axis(radicands: tuple[int, ...]) -> None:
    for index, radicand in enumerate(radicands):
        _require_surd_radicand(radicand, location=("radicands", str(index)))


def _require_scale_bits(scale_bits: int) -> None:
    if scale_bits < 1 or scale_bits > MAX_SURD_SCALE_BITS:
        raise OperationDomainValidationError(
            location=("scale_bits",),
            code="diophantine.scale_bits_out_of_range",
            message="scale_bits must lie in the admitted binary precision range",
        )


def _rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational(num=value.numerator, den=value.denominator)


def _interval(lower: Fraction, upper: Fraction) -> ClosedRationalInterval:
    return ClosedRationalInterval(lower=_rational(lower), upper=_rational(upper))


def _scaled_floor_row(multiplier: int, radicand: int) -> ScaledFloorValue:
    """Exact ``floor(n sqrt(d))`` from ``isqrt(d n^2)``; the radical is nonsquare."""

    square: int = radicand * multiplier * multiplier
    floor: int = isqrt(square)
    if floor * floor == square:
        # Only possible for a square radicand, which admission already rejects.
        raise OperationDomainValidationError(
            location=("radicand",),
            code="diophantine.scaled_floor_must_be_irrational",
            message="a scaled quadratic-surd floor requires a nonsquare radicand",
        )
    return ScaledFloorValue(
        multiplier=multiplier,
        radicand=radicand,
        floor=floor,
        ceiling=floor + 1,
        square_lower=floor * floor,
        square_upper=(floor + 1) * (floor + 1),
    )


def _distance_value(
    multiplier: int,
    radicand: int,
    scale_bits: int,
) -> NearestIntegerDistanceValue:
    """A certified enclosure of ``||n sqrt(d)||`` at ``2**-scale_bits``.

    ``sqrt(d n^2)`` lies strictly between ``floor`` and ``floor + 1``.  Scaling
    the integer square by ``4**scale_bits`` and taking its integer square root
    gives the scaled radical bracket ``[scaled_floor, scaled_floor + 1] / 2**s``
    with no floating point and no half-integer ambiguity, because the radical is
    irrational.
    """

    row = _scaled_floor_row(multiplier, radicand)
    scaled_square: int = radicand * multiplier * multiplier * 4**scale_bits
    scaled_floor: int = isqrt(scaled_square)
    if scaled_floor * scaled_floor == scaled_square:
        raise OperationDomainValidationError(
            location=("radicand",),
            code="diophantine.scaled_floor_must_be_irrational",
            message="a scaled quadratic-surd floor requires a nonsquare radicand",
        )
    scale: int = 2**scale_bits
    radical_lower = Fraction(scaled_floor, scale)
    radical_upper = Fraction(scaled_floor + 1, scale)

    floor_distance_lower = radical_lower - row.floor
    # The floor-branch distance stays strictly positive because the radical
    # exceeds its integer floor; the upper bound is a strict over-estimate.
    floor_distance_upper = radical_upper - row.floor
    ceiling_distance_lower = Fraction(row.ceiling) - radical_upper
    ceiling_distance_upper = Fraction(row.ceiling) - radical_lower

    # Strict separation is required: touching enclosures leave the branch
    # mathematically undecided, so they must not be read as a tie-break.
    side: Literal["FLOOR", "CEILING"]
    if floor_distance_upper < ceiling_distance_lower:
        side = "FLOOR"
        lower, upper = floor_distance_lower, floor_distance_upper
        upper_scaled = scaled_floor + 1 - row.floor * scale
    elif ceiling_distance_upper < floor_distance_lower:
        side = "CEILING"
        lower, upper = ceiling_distance_lower, ceiling_distance_upper
        upper_scaled = row.ceiling * scale - scaled_floor
    else:
        # Overlapping branches mean the requested precision cannot separate the
        # nearest integer; that is a caller error, never a guessed branch.
        raise OperationDomainValidationError(
            location=("scale_bits",),
            code="diophantine.nearest_integer_branch_unresolved",
            message=(
                "the requested scale_bits does not separate the nearest integer "
                "branch for this multiplier and radicand"
            ),
        )

    # The distance enclosure is nonnegative and ordered; both endpoints are
    # exact rationals, and neither endpoint is negative by construction.
    interval = bound_enclosure(
        _interval(max(lower, Fraction(0)), upper), label="nearest-integer distance"
    )
    return NearestIntegerDistanceValue(
        multiplier=multiplier,
        radicand=radicand,
        floor=row.floor,
        ceiling=row.ceiling,
        nearest_integer=row.floor if side == "FLOOR" else row.ceiling,
        side=side,
        scale_bits=scale_bits,
        distance_enclosure=interval,
        distance_upper_scaled=upper_scaled,
    )


def _product_enclosure(
    multiplier: int,
    factors: tuple[NearestIntegerDistanceValue, ...],
) -> ClosedRationalInterval:
    """Exact interval for ``n * prod_i ||n sqrt(d_i)||``.

    Every distance enclosure is nonnegative, so the outer multiplier scales
    both endpoints exactly and preserves the interval order.
    """

    lower = Fraction(multiplier)
    upper = Fraction(multiplier)
    for factor in factors:
        lower *= factor.distance_enclosure.lower.as_fraction()
        upper *= factor.distance_enclosure.upper.as_fraction()
    return bound_enclosure(_interval(lower, upper), label="simultaneous product")


def _range_admission(request: RangeProfileRequest) -> None:
    """Admit complete range expansion before allocating any result rows."""

    _require_request_multiplier(request.limit)
    radicand_count = len(request.radicands)
    multiplier_bits = request.limit.bit_length()
    radicand_bits = MAX_SURD_RADICAND.bit_length()
    scalar_bits = max(
        radicand_bits + 2 * multiplier_bits + request.scale_bits + 2,
        request.scale_bits + 2,
    )
    product_bits = multiplier_bits + radicand_count * (request.scale_bits + 2)
    work = request.limit * radicand_count * (request.scale_bits + scalar_bits)
    intermediate_bits = request.limit * radicand_count * scalar_bits
    row_bytes = radicand_count * (12 * scalar_bits + 512) + 4 * product_bits + 1_024
    output_bytes = request.limit * row_bytes + radicand_count * 32 + 256
    if work > MAX_SURD_RANGE_WORK:
        raise OperationResourceAdmissionError(
            location=("limit", "scale_bits"),
            code="diophantine.range_profile_work_bound",
            message="range expansion exceeds the admitted exact-work bound",
        )
    if intermediate_bits > MAX_SURD_RANGE_INTERMEDIATE_BITS:
        raise OperationResourceAdmissionError(
            location=("limit", "scale_bits"),
            code="diophantine.range_profile_intermediate_bound",
            message="range expansion exceeds the admitted intermediate-size bound",
        )
    if output_bytes > MAX_SURD_RANGE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("limit", "scale_bits"),
            code="diophantine.range_profile_output_bound",
            message="the complete serialized range exceeds the admitted output bound",
        )


def scaled_floor(request: ScaledFloorRequest) -> ScaledFloorValue:
    """Return one exact ``floor``/``ceiling`` value derived from integer squares."""

    _require_request_multiplier(request.multiplier)
    _require_surd_radicand(request.radicand, location=("radicand",))
    return _scaled_floor_row(request.multiplier, request.radicand)


def nearest_integer_distance(
    request: NearestIntegerDistanceRequest,
) -> NearestIntegerDistanceValue:
    """Certify the distance from ``n sqrt(d)`` to its nearest integer."""

    _require_request_multiplier(request.multiplier)
    _require_scale_bits(request.scale_bits)
    _require_surd_radicand(request.radicand, location=("radicand",))
    return _distance_value(request.multiplier, request.radicand, request.scale_bits)


def simultaneous_product(
    request: SimultaneousProductRequest,
) -> SimultaneousProductResult:
    """Certify ``n * prod_i ||n sqrt(d_i)||`` and every factor row."""

    _require_request_multiplier(request.multiplier)
    _require_scale_bits(request.scale_bits)
    _require_surd_axis(request.radicands)
    factors = tuple(
        _distance_value(request.multiplier, radicand, request.scale_bits)
        for radicand in request.radicands
    )
    return SimultaneousProductResult(
        multiplier=request.multiplier,
        radicands=request.radicands,
        scale_bits=request.scale_bits,
        factors=factors,
        product_enclosure=_product_enclosure(request.multiplier, factors),
    )


def _compute_range_profile(request: RangeProfileRequest) -> RangeProfileResult:
    rows = []
    for multiplier in range(1, request.limit + 1):
        try:
            factors = tuple(
                _distance_value(multiplier, radicand, request.scale_bits)
                for radicand in request.radicands
            )
        except OperationDomainValidationError as error:
            if error.errors()[0]["type"] != (
                "diophantine.nearest_integer_branch_unresolved"
            ):
                raise
            # Completeness is a contract: reject the whole range with the
            # first unresolvable row instead of returning a short profile.
            raise OperationDomainValidationError(
                location=("scale_bits",),
                code="diophantine.range_profile_unresolved_row",
                message=(
                    "the declared scale_bits cannot resolve the nearest-integer "
                    f"branch of row n={multiplier}; raise scale_bits or lower limit"
                ),
            ) from error
        rows.append(
            RangeProfileRow(
                multiplier=multiplier,
                factors=factors,
                product_enclosure=_product_enclosure(multiplier, factors),
            )
        )
    return RangeProfileResult(
        radicands=request.radicands,
        limit=request.limit,
        scale_bits=request.scale_bits,
        rows=tuple(rows),
    )


def range_profile(request: RangeProfileRequest) -> RangeProfileResult:
    """Return a complete certified row for every ``1 <= n <= limit``."""

    _require_scale_bits(request.scale_bits)
    _require_surd_axis(request.radicands)
    _range_admission(request)
    return _compute_range_profile(request)


def record_minima(request: RecordMinimaRequest) -> RecordMinimaResult:
    """Extract strict record minima, or report the first unseparated comparison.

    A new strict record requires its product enclosure to lie strictly below
    every prior incumbent enclosure.  When an enclosure overlaps the incumbent
    the comparison is mathematically undecided at this precision, so the result
    reports ``UNRESOLVED`` with both enclosures instead of guessing an order.
    """

    _require_scale_bits(request.scale_bits)
    _require_surd_axis(request.radicands)
    profile_request = RangeProfileRequest(
        radicands=request.radicands,
        limit=request.limit,
        scale_bits=request.scale_bits,
    )
    _range_admission(profile_request)
    profile = _compute_range_profile(
        RangeProfileRequest(
            radicands=request.radicands,
            limit=request.limit,
            scale_bits=request.scale_bits,
        )
    )
    records: list[RecordMinimumValue] = []
    incumbent: ClosedRationalInterval | None = None
    incumbent_lower = Fraction(0)
    for row in profile.rows:
        candidate = row.product_enclosure
        candidate_lower = candidate.lower.as_fraction()
        candidate_upper = candidate.upper.as_fraction()
        if incumbent is None:
            records.append(
                RecordMinimumValue(
                    multiplier=row.multiplier,
                    product_enclosure=candidate,
                    incumbent_enclosure=candidate,
                )
            )
            incumbent = candidate
            incumbent_lower = candidate_lower
            continue
        if candidate_upper < incumbent_lower:
            records.append(
                RecordMinimumValue(
                    multiplier=row.multiplier,
                    product_enclosure=candidate,
                    incumbent_enclosure=incumbent,
                )
            )
            incumbent = candidate
            incumbent_lower = candidate_lower
            continue
        incumbent_upper = incumbent.upper.as_fraction()
        if incumbent_upper <= candidate_lower:
            # The candidate is provably no smaller than the incumbent.  This
            # includes touching intervals: equality is not a strict record.
            continue
        # The intervals overlap (or the candidate reaches below the incumbent
        # while its lower endpoint does not establish a new record).  Either
        # ordering is possible at this precision, so make no record claim.
        if candidate_lower < incumbent_upper:
            return RecordMinimaResult(
                radicands=request.radicands,
                limit=request.limit,
                scale_bits=request.scale_bits,
                outcome="UNRESOLVED",
                records=tuple(records),
                unresolved_multiplier=row.multiplier,
                unresolved_product_enclosure=candidate,
                unresolved_incumbent_enclosure=incumbent,
            )
        raise AssertionError("record interval comparison was not classified")
    return RecordMinimaResult(
        radicands=request.radicands,
        limit=request.limit,
        scale_bits=request.scale_bits,
        outcome="COMPLETE",
        records=tuple(records),
        finite_argmin=records[-1].multiplier,
    )
