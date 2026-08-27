"""Contracts for bounded real-function point enclosures and their checker."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.analysis._arb import dyadic_endpoints
from jacobian.math.analysis._models import (
    MAX_DYADIC_MANTISSA_DIGITS,
    MAX_RATIONAL_DIGITS,
    ExactDyadic,
    _validation_error,
)

MAX_POINT_CHECK_DYADIC_EXPONENT = 8_192
MAX_POINT_CHECK_LOG_TERMS = 128
MAX_POINT_CHECK_FRACTION_BITS = 131_072
MAX_POINT_CHECK_OUTPUT_BYTES = 4_096


class RealUnaryFunction(StrEnum):
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    SIN = "SIN"
    COS = "COS"


class ArbPointEnclosureRequest(StrictModel):
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="validated-analysis rational",
        )
        return self


type PointEnclosureCheckOutcome = Literal["ACCEPTED", "REJECTED", "NON_RESULT"]


class ClaimedPointEnclosure(StrictModel):
    """One claimed enclosure of a real function value by exact dyadic endpoints.

    This is the domain-owned canonical value shared by the Arb producer and
    the independent checker, so a serialized claim crosses the consumer
    boundary unchanged. Endpoint order is deliberately not validated here: a
    reversed claim is a checkable mathematical statement, not an invalid one.
    """

    function: RealUnaryFunction = Field(
        description="Real function whose value the endpoints claim to enclose."
    )
    argument: CanonicalRational = Field(
        description="Exact reduced rational argument with at most 128 digits per component."
    )
    precision_bits: StrictInt = Field(
        ge=32,
        le=4096,
        description=(
            "Precision metadata retained from the source computation; it does "
            "not promise that an independent verification resolves the claim at "
            "that precision."
        ),
    )
    lower: ExactDyadic = Field(description="Claimed inclusive lower endpoint.")
    upper: ExactDyadic = Field(description="Claimed inclusive upper endpoint.")

    @model_validator(mode="after")
    def bound_claim_source(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="claimed point-enclosure rational",
        )
        return self


def _preflight_point_check_source(data: object) -> object:
    """Reject oversized raw source scalars before canonical integer parsing."""

    if not isinstance(data, dict):
        return data
    enclosure = data.get("enclosure")
    if isinstance(enclosure, ClaimedPointEnclosure):
        raw_components: tuple[object, ...] = (
            enclosure.argument.num,
            enclosure.argument.den,
        )
    elif isinstance(enclosure, dict):
        argument = enclosure.get("argument")
        if isinstance(argument, CanonicalRational):
            raw_components = (argument.num, argument.den)
        elif isinstance(argument, dict):
            raw_components = (argument.get("num"), argument.get("den"))
        else:
            raw_components = ()
    else:
        raw_components = ()
    if any(
        isinstance(component, str) and len(component.lstrip("-")) > MAX_RATIONAL_DIGITS
        for component in raw_components
    ):
        raise _validation_error(
            "point-enclosure checker raw rational component exceeds the "
            f"{MAX_RATIONAL_DIGITS}-digit bound"
        )
    return data


def _point_check_fraction_bound_bits(argument: CanonicalRational) -> int:
    """Bound LOG/SQRT intermediates, including claim comparisons and verification."""

    numerator, denominator = argument.as_integer_ratio()
    source_bits = max(abs(numerator).bit_length(), denominator.bit_length())
    transformed_bits = source_bits + 2
    odd_denominator_bits = (2 * MAX_POINT_CHECK_LOG_TERMS + 1).bit_length()

    # A common denominator for one upper atanh bound divides
    # b**(2*n-1) * lcm(1, 3, ..., 2*n-1) * (2*n+1) * (b*b-a*a).
    # Combining log(y) with k*log(2) adds the independent powers of 3 but
    # shares the odd-denominator factors.  The coefficient k has at most
    # source_bits.bit_length() bits after exact power-of-two range reduction.
    combined_log_bits = (
        (2 * MAX_POINT_CHECK_LOG_TERMS - 1) * (transformed_bits + 2)
        + 2 * transformed_bits
        + (MAX_POINT_CHECK_LOG_TERMS + 1) * odd_denominator_bits
        + source_bits.bit_length()
        + 8
    )
    endpoint_bits = 4 * MAX_DYADIC_MANTISSA_DIGITS + MAX_POINT_CHECK_DYADIC_EXPONENT
    log_comparison_bits = combined_log_bits + endpoint_bits + 32
    sqrt_comparison_bits = 2 * endpoint_bits + source_bits + 8
    return max(log_comparison_bits, sqrt_comparison_bits)


class PointEnclosureCheckRequest(StrictModel):
    """Check one claimed LOG or SQRT enclosure by exact independent verification.

    The claimed enclosure is one canonical ``ClaimedPointEnclosure`` accepted
    unchanged from its source. Rational components have at most 128 decimal
    digits. Claimed dyadic exponents must lie in -8192..8192; reversed or
    mathematically invalid intervals remain valid claims and produce typed
    checker outcomes. Only LOG and SQRT claims are admitted. LOG verification uses
    at most 128 terms per series, about 400 worst-case bits after range
    reduction, so tighter claims can produce NON_RESULT even when their
    retained precision metadata is larger.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_log_term_bound": MAX_POINT_CHECK_LOG_TERMS,
            "point_check_fraction_intermediate_bit_bound": (
                MAX_POINT_CHECK_FRACTION_BITS
            ),
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    enclosure: ClaimedPointEnclosure = Field(
        description=(
            "Canonical claimed enclosure retained verbatim from its source; "
            "only LOG and SQRT functions are admitted."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        return _preflight_point_check_source(canonicalize_json_containers(data))

    @model_validator(mode="after")
    def preflight_exact_checker(self) -> Self:
        if self.enclosure.function not in (
            RealUnaryFunction.LOG,
            RealUnaryFunction.SQRT,
        ):
            raise _validation_error(
                "point-enclosure checker verifies only LOG and SQRT claims"
            )
        if any(
            abs(endpoint.exponent) > MAX_POINT_CHECK_DYADIC_EXPONENT
            for endpoint in (self.enclosure.lower, self.enclosure.upper)
        ):
            raise _validation_error(
                "point-enclosure checker dyadic exponent exceeds the "
                f"+/-{MAX_POINT_CHECK_DYADIC_EXPONENT} bound"
            )
        if (
            _point_check_fraction_bound_bits(self.enclosure.argument)
            > MAX_POINT_CHECK_FRACTION_BITS
        ):
            raise _validation_error(
                "point-enclosure checker exact rational work exceeds the "
                f"{MAX_POINT_CHECK_FRACTION_BITS}-bit intermediate bound"
            )
        return self


class PointEnclosureCheckResult(StrictModel):
    """A source-bound outcome from the owner-private checker.

    ACCEPTED means the independently proved interval is contained in the
    claim. REJECTED covers an invalid real-domain or reversed claim, or a claim
    proved disjoint from the true value. NON_RESULT means the independent LOG
    enclosure still partially overlaps the claim after 128 series terms.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    enclosure: ClaimedPointEnclosure
    outcome: PointEnclosureCheckOutcome = Field(
        description=(
            "ACCEPTED when the independent enclosure is contained in the "
            "claim; REJECTED for invalid or provably excluding claims; "
            "NON_RESULT for unresolved partial overlap at the LOG term cap."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        return _preflight_point_check_source(canonicalize_json_containers(data))

    @classmethod
    def _from_kernel(
        cls,
        request: PointEnclosureCheckRequest,
        outcome: PointEnclosureCheckOutcome,
    ) -> Self:
        """Build an outcome after the admitted checker established it."""

        return cls.model_construct(enclosure=request.enclosure, outcome=outcome)


class ArbPointEnclosureResult(ArbPointEnclosureRequest):
    """A source-bound Arb ball enclosure of one real function value.

    Every outcome retains the request's function, argument, and precision;
    ``enclosure`` carries the canonical ``ClaimedPointEnclosure`` only when
    ``status`` is ``ENCLOSED``, and must restate that retained source.
    """

    status: Literal[
        "ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR", "OUTPUT_MAGNITUDE_EXCEEDED"
    ]
    enclosure: ClaimedPointEnclosure | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.enclosure is not None):
            raise _validation_error(
                "only an enclosed result may carry the point enclosure"
            )
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise _validation_error(
                "a non-enclosure cannot claim accuracy or exactness"
            )
        if enclosed:
            enclosure = self.enclosure
            assert enclosure is not None
            if (
                enclosure.function,
                enclosure.argument,
                enclosure.precision_bits,
            ) != (self.function, self.argument, self.precision_bits):
                raise _validation_error(
                    "the enclosure must restate the retained request"
                )
            if enclosure.lower.compare(enclosure.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.exact != (self.relative_accuracy_bits is None):
                raise _validation_error(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


def _point_enclosure(request: ArbPointEnclosureRequest) -> ArbPointEnclosureResult:
    """Compute one bounded Arb point enclosure in its owning family."""

    from flint import arb, ctx, fmpq

    numerator, denominator = request.argument.as_integer_ratio()
    with ctx.workprec(request.precision_bits):
        value = arb(fmpq(numerator, denominator))
        result = getattr(value, request.function.value.lower())()
        if not result.is_finite():
            return ArbPointEnclosureResult(
                function=request.function,
                argument=request.argument,
                precision_bits=request.precision_bits,
                status="NONFINITE",
                detail="Arb returned a non-finite ball; no enclosure conclusion is available.",
            )
        lower_mantissa, lower_exponent = result.lower().man_exp()
        upper_mantissa, upper_exponent = result.upper().man_exp()
        exact = bool(result.is_exact())
        endpoints = dyadic_endpoints(
            lower_mantissa, lower_exponent, upper_mantissa, upper_exponent
        )
    if endpoints is None:
        return ArbPointEnclosureResult(
            function=request.function,
            argument=request.argument,
            precision_bits=request.precision_bits,
            status="OUTPUT_MAGNITUDE_EXCEEDED",
            detail="Arb produced finite endpoints outside the interoperable dyadic exponent range.",
        )
    return ArbPointEnclosureResult(
        function=request.function,
        argument=request.argument,
        precision_bits=request.precision_bits,
        status="ENCLOSED",
        enclosure=ClaimedPointEnclosure(
            function=request.function,
            argument=request.argument,
            precision_bits=request.precision_bits,
            lower=endpoints[0],
            upper=endpoints[1],
        ),
        relative_accuracy_bits=None if exact else int(result.rel_accuracy_bits()),
        exact=exact,
        detail="Pinned Arb ball arithmetic returned an outward-rounded enclosure with exact dyadic endpoints.",
    )


def _check_point_enclosure(
    request: PointEnclosureCheckRequest,
) -> PointEnclosureCheckResult:
    """Verify one point-enclosure claim in its owning family."""

    from jacobian.math.analysis._point_enclosure_check import _verify_point_enclosure

    return PointEnclosureCheckResult._from_kernel(
        request, _verify_point_enclosure(request)
    )


__all__ = [
    "MAX_POINT_CHECK_DYADIC_EXPONENT",
    "MAX_POINT_CHECK_FRACTION_BITS",
    "MAX_POINT_CHECK_LOG_TERMS",
    "MAX_POINT_CHECK_OUTPUT_BYTES",
    "ArbPointEnclosureRequest",
    "ArbPointEnclosureResult",
    "ClaimedPointEnclosure",
    "PointEnclosureCheckOutcome",
    "PointEnclosureCheckRequest",
    "PointEnclosureCheckResult",
    "RealUnaryFunction",
]
