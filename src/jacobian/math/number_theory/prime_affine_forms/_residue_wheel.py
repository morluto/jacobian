"""Typed contracts and bounded kernels for prime-affine residue wheels."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.affine_forms.values import (
    AffineComponentInteger,
    AffineFormId,
)
from jacobian.math.number_theory.prime_affine_forms._interval import (
    IntervalEndpointInteger,
    _parse_interval,
)
from jacobian.math.number_theory.prime_affine_forms._models import (
    CompactPrime,
    PrimeTupleLocalSummary,
    _validation_error,
)
from jacobian.math.number_theory.prime_affine_forms.values import (
    MAX_AFFINE_FORMS,
    PrimeAffineTuple,
)

MAX_WHEEL_PRIMES = 64
MAX_WHEEL_LOCAL_RESIDUES = 32_768
MAX_WHEEL_RESIDUES = 8_192
MAX_WHEEL_RESULT_CELLS = 65_536
MAX_WHEEL_ENUMERATION_WORK = 300_000
MAX_COMPACT_WHEEL_ROOT_WORK = 250_000
MAX_COMPACT_WHEEL_SCALAR_DIGITS = 4_096
MAX_WHEEL_INTERVAL_LENGTH = 8_192

CompactWheelScalar = Annotated[
    CanonicalInteger,
    StringConstraints(max_length=MAX_COMPACT_WHEEL_SCALAR_DIGITS, strict=True),
]


class PrimeTupleResidueWheelRequest(StrictModel):
    """Construct a compact exact CRT wheel for a canonical finite prime set."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(
        max_length=MAX_WHEEL_PRIMES,
        description=(
            "Possibly empty tuple of distinct primes in strictly increasing order; "
            "the empty set returns the modulus-one identity wheel. This compact "
            "operation does not enumerate all combined residues."
        ),
    )


class PrimeTupleWheelResidueRow(StrictModel):
    residue: CompactWheelScalar
    components: tuple[StrictInt, ...] = Field(max_length=MAX_WHEEL_PRIMES)


class PrimeTupleResidueWheel(StrictModel):
    """Compact source-bound CRT wheel as a product of valid local residue sets."""

    source: PrimeAffineTuple
    primes: tuple[CompactPrime, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    local_rows: tuple[PrimeTupleLocalSummary, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    modulus: CompactWheelScalar
    valid_count: CompactWheelScalar

    @model_validator(mode="after")
    def require_compact_wheel_shape(self) -> Self:
        if tuple(row.prime for row in self.local_rows) != self.primes:
            raise _validation_error(
                "wheel local rows must align with its canonical primes"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleResidueWheelRequest,
        *,
        local_rows: tuple[PrimeTupleLocalSummary, ...],
        modulus: str,
        valid_count: str,
    ) -> Self:
        return cls(
            source=request.source,
            primes=request.primes,
            local_rows=local_rows,
            modulus=modulus,
            valid_count=valid_count,
        )


class PrimeTupleResidueWheelEnumerationRequest(StrictModel):
    """Materialize every residue of a supplied compact wheel under strict bounds."""

    wheel: PrimeTupleResidueWheel


class PrimeTupleResidueWheelEnumeration(StrictModel):
    """Complete explicit residue realization of one compact CRT wheel."""

    wheel: PrimeTupleResidueWheel
    residues: tuple[PrimeTupleWheelResidueRow, ...] = Field(
        max_length=MAX_WHEEL_RESIDUES
    )

    @model_validator(mode="after")
    def require_enumeration_shape(self) -> Self:
        result_cells = len(self.residues) + sum(
            len(row.components) for row in self.residues
        )
        if result_cells > MAX_WHEEL_RESULT_CELLS:
            raise _validation_error("wheel rows exceed the explicit result-cell bound")
        expected_count = parse_canonical_integer(self.wheel.valid_count)
        if len(self.residues) != expected_count:
            raise _validation_error(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        residues = tuple(parse_canonical_integer(row.residue) for row in self.residues)
        if residues != tuple(sorted(set(residues))):
            raise _validation_error(
                "wheel rows do not satisfy the complete CRT reconstruction invariant"
            )
        modulus = parse_canonical_integer(self.wheel.modulus)
        for row, residue in zip(self.residues, residues, strict=True):
            if not 0 <= residue < modulus or len(row.components) != len(
                self.wheel.primes
            ):
                raise _validation_error(
                    "wheel rows do not satisfy the complete CRT reconstruction invariant"
                )
            for prime, component in zip(self.wheel.primes, row.components, strict=True):
                if not 0 <= component < prime:
                    raise _validation_error(
                        "wheel components must be within their prime moduli"
                    )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleResidueWheelEnumerationRequest,
        *,
        residues: tuple[PrimeTupleWheelResidueRow, ...],
    ) -> Self:
        return cls(wheel=request.wheel, residues=residues)


class PrimeTupleWheelMembershipRequest(StrictModel):
    """Check one integer against an exact residue wheel supplied unchanged."""

    wheel: PrimeTupleResidueWheel
    value: AffineComponentInteger


class PrimeTupleWheelMembershipResult(StrictModel):
    wheel: PrimeTupleResidueWheel
    value: AffineComponentInteger
    canonical_residue: CompactWheelScalar
    components: tuple[StrictInt, ...] = Field(max_length=MAX_WHEEL_PRIMES)
    is_permitted: StrictBool
    first_excluded_prime: StrictInt | None = None
    vanishing_form_ids: tuple[AffineFormId, ...] = Field(max_length=MAX_AFFINE_FORMS)

    @model_validator(mode="after")
    def require_membership_shape(self) -> Self:
        if len(self.components) != len(self.wheel.primes):
            raise _validation_error(
                "wheel membership components must align with wheel primes"
            )
        if self.is_permitted != (self.first_excluded_prime is None):
            raise _validation_error(
                "wheel membership permission must agree with first exclusion"
            )
        if self.is_permitted and self.vanishing_form_ids:
            raise _validation_error("a permitted member cannot have vanishing form ids")
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleWheelMembershipRequest,
        *,
        canonical_residue: str,
        components: tuple[int, ...],
        is_permitted: bool,
        first_excluded_prime: int | None,
        vanishing_form_ids: tuple[str, ...],
    ) -> Self:
        return cls(
            wheel=request.wheel,
            value=request.value,
            canonical_residue=canonical_residue,
            components=components,
            is_permitted=is_permitted,
            first_excluded_prime=first_excluded_prime,
            vanishing_form_ids=vanishing_form_ids,
        )


class PrimeTupleIntervalResidueProfileRequest(StrictModel):
    """Enumerate local-wheel survivors on one bounded closed integer interval."""

    wheel: PrimeTupleResidueWheel
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger


class PrimeTupleIntervalResidueProfileResult(StrictModel):
    """Exact local survivors; this value does not assert actual primality."""

    wheel: PrimeTupleResidueWheel
    lower: IntervalEndpointInteger
    upper: IntervalEndpointInteger
    interval_size: StrictInt = Field(ge=1, le=MAX_WHEEL_INTERVAL_LENGTH)
    survivors: tuple[IntervalEndpointInteger, ...] = Field(
        max_length=MAX_WHEEL_INTERVAL_LENGTH
    )

    @model_validator(mode="after")
    def require_survivor_profile_shape(self) -> Self:
        lower, upper, interval_size = _parse_interval(self.lower, self.upper)
        actual = tuple(parse_canonical_integer(value) for value in self.survivors)
        if actual != tuple(sorted(set(actual))) or any(
            value < lower or value > upper for value in actual
        ):
            raise _validation_error(
                "survivors must be ordered distinct interval values"
            )
        if self.interval_size != interval_size:
            raise _validation_error("interval_size must equal upper-lower+1")
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PrimeTupleIntervalResidueProfileRequest,
        *,
        survivors: tuple[str, ...],
    ) -> Self:
        lower = parse_canonical_integer(request.lower)
        upper = parse_canonical_integer(request.upper)
        return cls(
            wheel=request.wheel,
            lower=request.lower,
            upper=request.upper,
            interval_size=upper - lower + 1,
            survivors=survivors,
        )


__all__ = [
    "MAX_WHEEL_INTERVAL_LENGTH",
    "MAX_WHEEL_PRIMES",
    "MAX_WHEEL_RESIDUES",
    "PrimeTupleIntervalResidueProfileRequest",
    "PrimeTupleIntervalResidueProfileResult",
    "PrimeTupleResidueWheel",
    "PrimeTupleResidueWheelEnumeration",
    "PrimeTupleResidueWheelEnumerationRequest",
    "PrimeTupleResidueWheelRequest",
    "PrimeTupleWheelMembershipRequest",
    "PrimeTupleWheelMembershipResult",
    "PrimeTupleWheelResidueRow",
]
