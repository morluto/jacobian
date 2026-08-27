"""Typed contracts and bounded kernels for prime-affine residue wheels."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.affine_forms.values import (
    MAX_AFFINE_COMPONENT_DIGITS,
    MAX_FORM_ID_LENGTH,
    AffineComponentInteger,
    AffineFormId,
)
from jacobian.math.prime_affine_forms._interval import (
    MAX_INTERVAL_EVALUATIONS,
    IntervalEndpointInteger,
    _parse_interval,
    require_bounded_affine_endpoints,
)
from jacobian.math.prime_affine_forms._models import (
    MAX_BATCH_PRIME,
    MAX_RESULT_CHARACTER_BUDGET,
    CompactPrime,
    PrimeTupleLocalSummary,
    _digits,
    _require_prime_set,
    _source_character_upper_bound,
    _summary_character_upper_bound,
    _validation_error,
)
from jacobian.math.prime_affine_forms.values import (
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

    @model_validator(mode="after")
    def require_bounded_compact_wheel(self) -> Self:
        _require_prime_set(self.primes, maximum=MAX_BATCH_PRIME)
        root_cells = self.source.form_count * len(self.primes)
        root_work = 6 * root_cells
        if root_work > MAX_COMPACT_WHEEL_ROOT_WORK:
            raise _validation_error(
                f"compact wheel computation needs {root_work} root "
                f"steps, exceeding {MAX_COMPACT_WHEEL_ROOT_WORK}"
            )
        modulus_digits = sum(_digits(prime) for prime in self.primes) or 1
        if modulus_digits > MAX_COMPACT_WHEEL_SCALAR_DIGITS:
            raise _validation_error(
                "compact wheel modulus exceeds the conservative exact scalar "
                f"digit bound {MAX_COMPACT_WHEEL_SCALAR_DIGITS}"
            )
        estimated_characters = (
            _source_character_upper_bound(self.source)
            + sum(
                _summary_character_upper_bound(self.source, prime)
                for prime in self.primes
            )
            + 2 * modulus_digits
            + 128
        )
        if estimated_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise _validation_error(
                "compact wheel exceeds the conservative serialized bound"
            )
        return self


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

    @model_validator(mode="after")
    def require_bounded_wheel_enumeration(self) -> Self:
        _require_verified_residue_wheel(self.wheel)
        local_residue_rows = sum(self.wheel.primes)
        if local_residue_rows > MAX_WHEEL_LOCAL_RESIDUES:
            raise _validation_error(
                f"wheel local residue enumeration {local_residue_rows} exceeds "
                f"{MAX_WHEEL_LOCAL_RESIDUES}"
            )
        result_count = parse_canonical_integer(self.wheel.valid_count)
        if result_count > MAX_WHEEL_RESIDUES:
            raise _validation_error(
                f"wheel has {result_count} valid residues, exceeding "
                f"{MAX_WHEEL_RESIDUES}"
            )
        result_cells = result_count * (len(self.wheel.primes) + 1)
        if result_cells > MAX_WHEEL_RESULT_CELLS:
            raise _validation_error(
                f"wheel result needs {result_cells} cells, exceeding "
                f"{MAX_WHEEL_RESULT_CELLS}"
            )
        root_cells = self.wheel.source.form_count * len(self.wheel.primes)
        enumeration_work = (
            result_count * len(self.wheel.primes) + local_residue_rows + root_cells
        )
        if enumeration_work > MAX_WHEEL_ENUMERATION_WORK:
            raise _validation_error(
                f"wheel enumeration needs {enumeration_work} bounded "
                f"steps, exceeding {MAX_WHEEL_ENUMERATION_WORK}"
            )
        modulus_digits = _digits(self.wheel.modulus)
        component_digits = sum(_digits(prime) for prime in self.wheel.primes)
        serialized_characters = (
            len(self.wheel.model_dump_json())
            + result_count
            * (modulus_digits + component_digits + 4 * len(self.wheel.primes) + 64)
            + 128
        )
        if serialized_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise _validation_error(
                "wheel enumeration exceeds the conservative serialized bound"
            )
        return self


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

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        _require_verified_residue_wheel(self.wheel)
        if _digits(self.value) > MAX_AFFINE_COMPONENT_DIGITS:
            raise _validation_error(
                f"membership value must have at most {MAX_AFFINE_COMPONENT_DIGITS} digits"
            )
        result_characters = (
            len(self.wheel.model_dump_json())
            + len(self.value)
            + _digits(self.wheel.modulus)
            + sum(_digits(prime) for prime in self.wheel.primes)
            + MAX_FORM_ID_LENGTH * self.wheel.source.form_count
            + 256
        )
        if result_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise _validation_error(
                "wheel membership result exceeds the conservative serialized bound"
            )
        return self


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

    @model_validator(mode="after")
    def require_bounded_survivor_profile(self) -> Self:
        _require_verified_residue_wheel(self.wheel)
        require_bounded_affine_endpoints(
            self.wheel.source, self.lower, self.upper, label="interval"
        )
        _, _, interval_size = _parse_interval(self.lower, self.upper)
        if interval_size > MAX_WHEEL_INTERVAL_LENGTH:
            raise _validation_error(
                f"wheel interval length {interval_size} exceeds "
                f"{MAX_WHEEL_INTERVAL_LENGTH}"
            )
        membership_checks = interval_size * max(1, len(self.wheel.primes))
        if membership_checks > MAX_INTERVAL_EVALUATIONS:
            raise _validation_error(
                f"wheel interval profile needs {membership_checks} modular checks, "
                f"exceeding {MAX_INTERVAL_EVALUATIONS}"
            )
        endpoint_digits = max(_digits(self.lower), _digits(self.upper))
        result_characters = (
            len(self.wheel.model_dump_json())
            + interval_size * (endpoint_digits + 4)
            + 192
        )
        if result_characters > MAX_RESULT_CHARACTER_BUDGET:
            raise _validation_error(
                "wheel interval profile exceeds the conservative serialized bound"
            )
        return self


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


def verify_residue_wheel(result: PrimeTupleResidueWheel) -> bool:
    """Verify an independently supplied compact residue-wheel claim."""

    from jacobian.math.prime_affine_forms._operations import compute_residue_wheel

    request = PrimeTupleResidueWheelRequest(source=result.source, primes=result.primes)
    return result == compute_residue_wheel(request)


def _require_verified_residue_wheel(wheel: PrimeTupleResidueWheel) -> None:
    """Admit an independently supplied wheel before a consumer trusts it."""

    if not verify_residue_wheel(wheel):
        raise _validation_error(
            "wheel must equal the compact residue wheel for its source and primes"
        )


def verify_residue_wheel_enumeration(
    result: PrimeTupleResidueWheelEnumeration,
) -> bool:
    """Verify an independently supplied explicit residue-wheel enumeration."""

    from jacobian.math.prime_affine_forms._operations import (
        compute_residue_wheel_enumeration,
    )

    return result == compute_residue_wheel_enumeration(
        PrimeTupleResidueWheelEnumerationRequest(wheel=result.wheel)
    )


def verify_wheel_membership_result(result: PrimeTupleWheelMembershipResult) -> bool:
    """Verify an independently supplied wheel-membership claim."""

    from jacobian.math.prime_affine_forms._operations import compute_wheel_membership

    return result == compute_wheel_membership(
        PrimeTupleWheelMembershipRequest(wheel=result.wheel, value=result.value)
    )


def verify_interval_residue_profile_result(
    result: PrimeTupleIntervalResidueProfileResult,
) -> bool:
    """Verify an independently supplied complete local survivor profile."""

    from jacobian.math.prime_affine_forms._operations import (
        compute_interval_residue_profile,
    )

    return result == compute_interval_residue_profile(
        PrimeTupleIntervalResidueProfileRequest(
            wheel=result.wheel, lower=result.lower, upper=result.upper
        )
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
