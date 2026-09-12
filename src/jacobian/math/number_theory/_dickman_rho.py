"""Certified piecewise polynomial enclosures for the Dickman rho function.

Write ``u = n + 1/2 + t/2`` on a unit piece, with ``-1 <= t <= 1``.  If the
preceding central polynomial is ``a(t)`` and the new one is ``b(t)``,
coefficient comparison in ``(2*n+1+t)b'(t) = -a(t)`` gives

``(2*n+1)(k+1)b[k+1] + k b[k] = -a[k]``.

After enforcing this through degree ``d-1``, the residual is the single term
``(d*b[d] + a[d])t**d``.  Since ``2*n+1+t >= 2*n`` on the whole interval,
variation of constants bounds its contribution by ``|r|/(n*(d+1))``.  A
preceding uniform error ``E`` contributes at most ``E`` at the shared endpoint
and ``E/n`` through the delay term.  Their sum is the propagated remainder
used below.  Dyadic outward rounding of every central coefficient is accounted
for separately in the reported pointwise width.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis._models import ExactDyadic

MAX_DICKMAN_ENDPOINT = 8
MAX_DICKMAN_DEGREE = 256
MIN_DICKMAN_PRECISION_BITS = 32
MAX_DICKMAN_PRECISION_BITS = 512
DEFAULT_DICKMAN_PRECISION_BITS = 128
MAX_DICKMAN_ENDPOINT_DIGITS = 4
MAX_DICKMAN_RESULT_COEFFICIENTS = MAX_DICKMAN_ENDPOINT * (MAX_DICKMAN_DEGREE + 1)
MAX_DICKMAN_WORK_UNITS = 50_000_000
# If the preceding coefficients are bounded by A, the centered recurrence has
# |b[0]| <= (d+1)A and |b[k+1]| <= A/2 + |b[k]|/2.  Across at most eight
# pieces this gives a coefficient magnitude below 2**65, which is retained in
# the exact intermediate and result envelope below.
MAX_DICKMAN_INTERMEDIATE_BITS = MAX_DICKMAN_PRECISION_BITS + MAX_DICKMAN_DEGREE + 65
MAX_DICKMAN_RESULT_BITS = (
    2 * MAX_DICKMAN_RESULT_COEFFICIENTS * MAX_DICKMAN_INTERMEDIATE_BITS
)


class DickmanRhoPiecewiseEnclosureParameters(StrictModel):
    endpoint: CanonicalRational = Field(
        description=(
            "Requested nonnegative endpoint U. The returned partition covers "
            "the complete integer interval [0, ceil(U)]."
        )
    )
    target_width: ExactDyadic = Field(
        description=(
            "Positive pointwise enclosure width required on every returned "
            "unit interval."
        )
    )
    precision_bits: StrictInt = Field(
        default=DEFAULT_DICKMAN_PRECISION_BITS,
        ge=MIN_DICKMAN_PRECISION_BITS,
        le=MAX_DICKMAN_PRECISION_BITS,
        description=(
            "Binary precision used for outward coefficient and remainder "
            "rounding; increasing it permits narrower certified widths."
        ),
    )

    @model_validator(mode="after")
    def require_admitted_range(self) -> Self:
        endpoint = self.endpoint.as_fraction()
        if endpoint < 0 or endpoint > MAX_DICKMAN_ENDPOINT:
            raise ValueError("Dickman endpoint must lie in [0, 8]")
        if self.target_width.as_fraction() <= 0:
            raise ValueError("target width must be positive")
        return self


# The catalog keeps the established request symbol while native consumers use
# the same canonical domain-owned parameter value without a transport-only
# model boundary.
DickmanRhoPiecewiseEnclosureRequest = DickmanRhoPiecewiseEnclosureParameters


class DyadicCoefficientBall(StrictModel):
    lower: ExactDyadic
    upper: ExactDyadic

    @model_validator(mode="after")
    def require_ordered(self) -> Self:
        if self.lower.compare(self.upper) > 0:
            raise ValueError("coefficient ball endpoints must be ordered")
        return self


class DickmanRhoAffineAxis(StrictModel):
    """The canonical affine coordinate used by one unit interval."""

    source_axis: Literal["u"] = Field(
        default="u",
        description="Dickman rho's source coordinate.",
    )
    polynomial_axis: Literal["t"] = Field(
        default="t",
        description="Polynomial coordinate, constrained to -1 <= t <= 1.",
    )
    center: CanonicalRational = Field(
        description="The interval midpoint in the source coordinate.",
    )
    scale: CanonicalRational = Field(
        description="The source half-width; every unit interval uses 1/2.",
    )

    @model_validator(mode="after")
    def require_unit_centered_map(self) -> Self:
        if self.scale.as_fraction() != Fraction(1, 2):
            raise ValueError("Dickman affine axes must use source scale 1/2")
        return self


class DickmanRhoAffinePiece(StrictModel):
    lower: StrictInt = Field(ge=0, le=MAX_DICKMAN_ENDPOINT)
    upper: StrictInt = Field(ge=1, le=MAX_DICKMAN_ENDPOINT)
    axis: DickmanRhoAffineAxis
    coefficients: tuple[DyadicCoefficientBall, ...] = Field(
        min_length=1, max_length=MAX_DICKMAN_DEGREE + 1
    )
    uniform_remainder: ExactDyadic

    @model_validator(mode="after")
    def require_piece(self) -> Self:
        if self.upper != self.lower + 1:
            raise ValueError("Dickman pieces must be consecutive unit intervals")
        if self.axis.center.as_fraction() != Fraction(2 * self.lower + 1, 2):
            raise ValueError("Dickman axis center must be the interval midpoint")
        if self.uniform_remainder.as_fraction() < 0:
            raise ValueError("uniform remainder must be nonnegative")
        return self


class DickmanRhoPiecewiseEnclosureResult(StrictModel):
    source: DickmanRhoPiecewiseEnclosureParameters
    degree: StrictInt = Field(ge=0, le=MAX_DICKMAN_DEGREE)
    pieces: tuple[DickmanRhoAffinePiece, ...] = Field(max_length=MAX_DICKMAN_ENDPOINT)

    @model_validator(mode="after")
    def require_partition_and_width(self) -> Self:
        endpoint = self.source.endpoint.as_fraction()
        if endpoint == 0:
            if self.pieces:
                raise ValueError("the zero interval has no Dickman pieces")
            return self
        if not self.pieces or self.pieces[0].lower != 0:
            raise ValueError("Dickman pieces must start at zero")
        cursor = 0
        if (
            len(self.pieces)
            != (endpoint.numerator + endpoint.denominator - 1) // endpoint.denominator
        ):
            raise ValueError("Dickman piece count does not match the source")
        target = self.source.target_width.as_fraction()
        for piece in self.pieces:
            if piece.lower != cursor:
                raise ValueError("Dickman pieces must form a contiguous partition")
            cursor = piece.upper
            if len(piece.coefficients) != self.degree + 1:
                raise ValueError("every Dickman piece must use the declared degree")
            width = 2 * piece.uniform_remainder.as_fraction() + sum(
                coefficient.upper.as_fraction() - coefficient.lower.as_fraction()
                for coefficient in piece.coefficients
            )
            if width > target:
                raise ValueError("Dickman piece exceeds the requested uniform width")
        expected = (
            endpoint.numerator + endpoint.denominator - 1
        ) // endpoint.denominator
        if cursor != expected:
            raise ValueError("Dickman pieces must end at ceil(U)")
        return self


def _dyadic_pair(mantissa: int, exponent: int) -> ExactDyadic:
    if mantissa == 0:
        return ExactDyadic(mantissa=0, exponent=0)
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return ExactDyadic(mantissa=mantissa, exponent=exponent)


def _dyadic_ball(value: Fraction, precision_bits: int) -> DyadicCoefficientBall:
    scale = 1 << precision_bits
    scaled_num = value.numerator * scale
    lower = scaled_num // value.denominator
    upper = -(-scaled_num // value.denominator)
    return DyadicCoefficientBall(
        lower=_dyadic_pair(lower, -precision_bits),
        upper=_dyadic_pair(upper, -precision_bits),
    )


def _dyadic_upper(value: Fraction, precision_bits: int) -> ExactDyadic:
    scale = 1 << precision_bits
    mantissa = -(-(value.numerator * scale) // value.denominator)
    return _dyadic_pair(mantissa, -precision_bits)


def _next_coefficients(previous: list[Fraction], n: int) -> list[Fraction]:
    degree = len(previous) - 1
    current = [Fraction(), *(Fraction() for _ in range(degree))]
    for k in range(degree):
        current[k + 1] = (-previous[k] - k * current[k]) / ((k + 1) * (2 * n + 1))
    previous_endpoint = sum(previous, Fraction())
    current[0] = previous_endpoint - sum(
        (-1) ** k * coefficient for k, coefficient in enumerate(current)
    )
    return current


def _central_pieces(
    endpoint: Fraction, degree: int
) -> list[tuple[Fraction, Fraction, list[Fraction], Fraction]]:
    if endpoint == 0:
        return []
    pieces: list[tuple[Fraction, Fraction, list[Fraction], Fraction]] = []
    coefficients = [Fraction(1), *(Fraction() for _ in range(degree))]
    pieces.append((Fraction(), Fraction(1), coefficients, Fraction()))
    interval_count = (
        endpoint.numerator + endpoint.denominator - 1
    ) // endpoint.denominator
    if interval_count == 1:
        return pieces

    previous = coefficients
    previous_remainder = Fraction()
    n = 1
    while n < interval_count:
        current = _next_coefficients(previous, n)
        residual = abs(degree * current[degree] + previous[degree])
        remainder = (
            previous_remainder + previous_remainder / n + residual / (n * (degree + 1))
        )
        pieces.append((Fraction(n), Fraction(n + 1), current, remainder))
        previous = current
        previous_remainder = remainder
        n += 1
    return pieces


def _piece_width(
    coefficients: list[Fraction], remainder: Fraction, precision_bits: int
) -> Fraction:
    rounding = sum(
        ball.upper.as_fraction() - ball.lower.as_fraction()
        for ball in (
            _dyadic_ball(coefficient, precision_bits) for coefficient in coefficients
        )
    )
    return rounding + 2 * _dyadic_upper(remainder, precision_bits).as_fraction()


def _admit_request(request: DickmanRhoPiecewiseEnclosureParameters) -> int:
    """Compute the complete semantic envelope before recurrence expansion."""

    endpoint = request.endpoint.as_fraction()
    interval_count = (
        endpoint.numerator + endpoint.denominator - 1
    ) // endpoint.denominator
    if (
        canonical_rational_component_digits(request.endpoint)
        > MAX_DICKMAN_ENDPOINT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("endpoint",),
            code="number_theory.dickman_rho.endpoint_representation",
            message="endpoint components exceed the admitted exact preflight bound",
        )
    if interval_count > 1 and request.target_width.as_fraction() < Fraction(
        1, 1 << request.precision_bits
    ):
        raise OperationResourceAdmissionError(
            location=("target_width", "precision_bits"),
            code="number_theory.dickman_rho.precision_floor",
            message=(
                "target width is below the unavoidable dyadic coefficient "
                "rounding floor for the requested precision"
            ),
        )
    candidate_work = interval_count * sum(
        degree * (request.precision_bits + degree + 64)
        for degree in range(8, MAX_DICKMAN_DEGREE + 1, 8)
    )
    if candidate_work > MAX_DICKMAN_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("endpoint", "precision_bits"),
            code="number_theory.dickman_rho.work_bound",
            message="Dickman recurrence work exceeds the admitted exact envelope",
        )
    result_bits = (
        2
        * interval_count
        * (MAX_DICKMAN_DEGREE + 1)
        * (request.precision_bits + MAX_DICKMAN_DEGREE + 64)
    )
    if result_bits > MAX_DICKMAN_RESULT_BITS:
        raise OperationResourceAdmissionError(
            location=("precision_bits",),
            code="number_theory.dickman_rho.result_bound",
            message="Dickman coefficient growth exceeds the admitted exact envelope",
        )
    return interval_count


def _run_dickman_rho_piecewise_enclosure(
    request: DickmanRhoPiecewiseEnclosureParameters,
) -> DickmanRhoPiecewiseEnclosureResult:
    endpoint = request.endpoint.as_fraction()
    target = request.target_width.as_fraction()
    _admit_request(request)
    for degree in range(8, MAX_DICKMAN_DEGREE + 1, 8):
        central = _central_pieces(endpoint, degree)
        if all(
            _piece_width(coefficients, remainder, request.precision_bits) <= target
            for _, _, coefficients, remainder in central
        ):
            return DickmanRhoPiecewiseEnclosureResult(
                source=request,
                degree=degree,
                pieces=tuple(
                    DickmanRhoAffinePiece(
                        lower=int(lower),
                        upper=int(upper),
                        axis=DickmanRhoAffineAxis(
                            center=CanonicalRational.from_fraction(
                                Fraction(lower + upper, 2)
                            ),
                            scale=CanonicalRational.from_fraction(Fraction(1, 2)),
                        ),
                        coefficients=tuple(
                            _dyadic_ball(coefficient, request.precision_bits)
                            for coefficient in coefficients
                        ),
                        uniform_remainder=_dyadic_upper(
                            remainder, request.precision_bits
                        ),
                    )
                    for lower, upper, coefficients, remainder in central
                ),
            )
    raise OperationResourceAdmissionError(
        location=("target_width",),
        code="number_theory.dickman_rho.degree_bound",
        message="the proved degree-256 enclosure does not meet the requested width",
    )


def dickman_rho_piecewise_enclosure(
    endpoint: CanonicalRational,
    target_width: ExactDyadic,
    *,
    precision_bits: int = DEFAULT_DICKMAN_PRECISION_BITS,
) -> DickmanRhoPiecewiseEnclosureResult:
    """Return a certified centered polynomial enclosure through ``endpoint``."""

    request = DickmanRhoPiecewiseEnclosureParameters(
        endpoint=endpoint,
        target_width=target_width,
        precision_bits=precision_bits,
    )
    return _run_dickman_rho_piecewise_enclosure(request)


DICKMAN_RHO_OPERATIONS = (
    MathTool(
        operation_id="number_theory.dickman_rho.piecewise_enclosure.compute",
        title="Enclose the Dickman rho function piecewise",
        description=(
            "Return dyadic coefficient balls and a uniform remainder for each "
            "integer unit interval through ceil(U), with u=center+t/2 and "
            "-1<=t<=1. The bound follows the delay recurrence and an explicit "
            "integrated residual."
        ),
        request_type=DickmanRhoPiecewiseEnclosureRequest,
        result_type=DickmanRhoPiecewiseEnclosureResult,
        run=_run_dickman_rho_piecewise_enclosure,
        tags=("number-theory", "Dickman", "rho", "enclosure", "validated"),
        examples=(
            OperationExample(
                name="through_second_interval",
                description=(
                    "Enclose rho through u=2; U must be nonnegative and the "
                    "target width must fit the bounded degree and precision."
                ),
                input={
                    "endpoint": {"num": "2", "den": "1"},
                    "target_width": {"mantissa": "1", "exponent": -5},
                    "precision_bits": 128,
                },
            ),
        ),
    ),
)


__all__ = [
    "DICKMAN_RHO_OPERATIONS",
    "DickmanRhoAffineAxis",
    "DickmanRhoAffinePiece",
    "DickmanRhoPiecewiseEnclosureParameters",
    "DickmanRhoPiecewiseEnclosureRequest",
    "DickmanRhoPiecewiseEnclosureResult",
    "DyadicCoefficientBall",
    "dickman_rho_piecewise_enclosure",
]
