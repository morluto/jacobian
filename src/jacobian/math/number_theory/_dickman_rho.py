"""Certified piecewise polynomial enclosures for the Dickman rho function.

Write ``u = n + x`` on a unit piece.  If the preceding central polynomial is
``a(x)`` and the new one is ``b(x)``, coefficient comparison in
``(n+x)b'(x) = -a(x)`` gives

``n(k+1)b[k+1] + k b[k] = -a[k]``.

After enforcing this through degree ``d-1``, the residual is the single term
``(d*b[d] + a[d])x**d``.  Variation of constants bounds its contribution on
``[0,h]`` by ``|r|h**(d+1)/(n(d+1))``.  A preceding uniform error ``E``
contributes at most ``E`` at the shared endpoint and ``Eh/n`` through the
delay term.  Their sum is the propagated remainder used below.  Dyadic
outward rounding of every central coefficient is accounted for separately in
the reported pointwise width.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis._models import ExactDyadic

MAX_DICKMAN_ENDPOINT = 8
MAX_DICKMAN_DEGREE = 256
COEFFICIENT_BITS = 128


class DickmanRhoPiecewiseEnclosureRequest(StrictModel):
    endpoint: CanonicalRational
    target_width: ExactDyadic

    @model_validator(mode="after")
    def require_admitted_range(self) -> Self:
        endpoint = self.endpoint.as_fraction()
        if endpoint < 0 or endpoint > MAX_DICKMAN_ENDPOINT:
            raise ValueError("Dickman endpoint must lie in [0, 8]")
        if self.target_width.as_fraction() <= 0:
            raise ValueError("target width must be positive")
        return self


class DyadicCoefficientBall(StrictModel):
    lower: ExactDyadic
    upper: ExactDyadic

    @model_validator(mode="after")
    def require_ordered(self) -> Self:
        if self.lower.compare(self.upper) > 0:
            raise ValueError("coefficient ball endpoints must be ordered")
        return self


class DickmanRhoAffinePiece(StrictModel):
    lower: CanonicalRational
    upper: CanonicalRational
    coefficients: tuple[DyadicCoefficientBall, ...] = Field(
        min_length=1, max_length=MAX_DICKMAN_DEGREE + 1
    )
    uniform_remainder: ExactDyadic

    @model_validator(mode="after")
    def require_piece(self) -> Self:
        if self.lower.as_fraction() >= self.upper.as_fraction():
            raise ValueError("Dickman pieces must have positive length")
        if self.uniform_remainder.as_fraction() < 0:
            raise ValueError("uniform remainder must be nonnegative")
        return self


class DickmanRhoPiecewiseEnclosureResult(StrictModel):
    source: DickmanRhoPiecewiseEnclosureRequest
    degree: StrictInt = Field(ge=0, le=MAX_DICKMAN_DEGREE)
    pieces: tuple[DickmanRhoAffinePiece, ...] = Field(max_length=MAX_DICKMAN_ENDPOINT)

    @model_validator(mode="after")
    def require_partition_and_width(self) -> Self:
        endpoint = self.source.endpoint.as_fraction()
        if endpoint == 0:
            if self.pieces:
                raise ValueError("the zero interval has no Dickman pieces")
            return self
        if not self.pieces or self.pieces[0].lower.as_fraction() != 0:
            raise ValueError("Dickman pieces must start at zero")
        cursor = Fraction()
        target = self.source.target_width.as_fraction()
        expected = _central_pieces(endpoint, self.degree)
        if len(expected) != len(self.pieces):
            raise ValueError("Dickman piece count does not match the source")
        for piece, (lower, upper, coefficients, remainder) in zip(
            self.pieces, expected, strict=True
        ):
            if piece.lower.as_fraction() != cursor:
                raise ValueError("Dickman pieces must form a contiguous partition")
            cursor = piece.upper.as_fraction()
            if piece.lower.as_fraction() != lower or piece.upper.as_fraction() != upper:
                raise ValueError("Dickman piece endpoints do not match the source")
            if len(piece.coefficients) != self.degree + 1:
                raise ValueError("every Dickman piece must use the declared degree")
            if any(
                ball.lower.as_fraction() > coefficient
                or ball.upper.as_fraction() < coefficient
                for ball, coefficient in zip(
                    piece.coefficients, coefficients, strict=True
                )
            ):
                raise ValueError("coefficient balls must enclose the recurrence")
            if piece.uniform_remainder.as_fraction() < remainder:
                raise ValueError("uniform remainder must enclose the proved residual")
            width = 2 * piece.uniform_remainder.as_fraction() + sum(
                coefficient.upper.as_fraction() - coefficient.lower.as_fraction()
                for coefficient in piece.coefficients
            )
            if width > target:
                raise ValueError("Dickman piece exceeds the requested uniform width")
        if cursor != endpoint:
            raise ValueError("Dickman pieces must end at the requested endpoint")
        return self


def _dyadic_pair(mantissa: int, exponent: int) -> ExactDyadic:
    if mantissa == 0:
        return ExactDyadic(mantissa=0, exponent=0)
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return ExactDyadic(mantissa=mantissa, exponent=exponent)


def _dyadic_ball(value: Fraction) -> DyadicCoefficientBall:
    scale = 1 << COEFFICIENT_BITS
    scaled_num = value.numerator * scale
    lower = scaled_num // value.denominator
    upper = -(-scaled_num // value.denominator)
    return DyadicCoefficientBall(
        lower=_dyadic_pair(lower, -COEFFICIENT_BITS),
        upper=_dyadic_pair(upper, -COEFFICIENT_BITS),
    )


def _dyadic_upper(value: Fraction) -> ExactDyadic:
    scale = 1 << COEFFICIENT_BITS
    mantissa = -(-(value.numerator * scale) // value.denominator)
    return _dyadic_pair(mantissa, -COEFFICIENT_BITS)


def _next_coefficients(previous: list[Fraction], n: int) -> list[Fraction]:
    degree = len(previous) - 1
    current = [sum(previous, Fraction()), *(Fraction() for _ in range(degree))]
    for k in range(degree):
        current[k + 1] = (-previous[k] - k * current[k]) / ((k + 1) * n)
    return current


def _central_pieces(
    endpoint: Fraction, degree: int
) -> list[tuple[Fraction, Fraction, list[Fraction], Fraction]]:
    if endpoint == 0:
        return []
    pieces: list[tuple[Fraction, Fraction, list[Fraction], Fraction]] = []
    first_upper = min(endpoint, Fraction(1))
    coefficients = [Fraction(1), *(Fraction() for _ in range(degree))]
    pieces.append((Fraction(), first_upper, coefficients, Fraction()))
    if endpoint <= 1:
        return pieces

    previous = coefficients
    previous_remainder = Fraction()
    n = 1
    while Fraction(n) < endpoint:
        height = min(Fraction(1), endpoint - n)
        current = _next_coefficients(previous, n)
        residual = abs(degree * current[degree] + previous[degree])
        remainder = (
            previous_remainder
            + previous_remainder * height / n
            + residual * height ** (degree + 1) / (n * (degree + 1))
        )
        pieces.append((Fraction(n), Fraction(n) + height, current, remainder))
        if height < 1:
            break
        previous = current
        previous_remainder = remainder
        n += 1
    return pieces


def _piece_width(coefficients: list[Fraction], remainder: Fraction) -> Fraction:
    rounding = sum(
        ball.upper.as_fraction() - ball.lower.as_fraction()
        for ball in map(_dyadic_ball, coefficients)
    )
    return rounding + 2 * _dyadic_upper(remainder).as_fraction()


def dickman_rho_piecewise_enclosure(
    request: DickmanRhoPiecewiseEnclosureRequest,
) -> DickmanRhoPiecewiseEnclosureResult:
    endpoint = request.endpoint.as_fraction()
    target = request.target_width.as_fraction()
    for degree in range(8, MAX_DICKMAN_DEGREE + 1, 8):
        central = _central_pieces(endpoint, degree)
        if all(
            _piece_width(coefficients, remainder) <= target
            for _, _, coefficients, remainder in central
        ):
            return DickmanRhoPiecewiseEnclosureResult(
                source=request,
                degree=degree,
                pieces=tuple(
                    DickmanRhoAffinePiece(
                        lower=CanonicalRational.from_fraction(lower),
                        upper=CanonicalRational.from_fraction(upper),
                        coefficients=tuple(map(_dyadic_ball, coefficients)),
                        uniform_remainder=_dyadic_upper(remainder),
                    )
                    for lower, upper, coefficients, remainder in central
                ),
            )
    raise OperationResourceAdmissionError(
        location=("target_width",),
        code="number_theory.dickman_rho.degree_bound",
        message="the proved degree-256 enclosure does not meet the requested width",
    )


DICKMAN_RHO_OPERATIONS = (
    MathTool(
        operation_id="number_theory.dickman_rho.piecewise_enclosure.compute",
        title="Enclose the Dickman rho function piecewise",
        description=(
            "Return dyadic coefficient balls and a uniform remainder for each "
            "affine unit-interval piece of the Dickman rho function. The bound "
            "follows the delay recurrence and an explicit integrated residual."
        ),
        request_type=DickmanRhoPiecewiseEnclosureRequest,
        result_type=DickmanRhoPiecewiseEnclosureResult,
        run=dickman_rho_piecewise_enclosure,
        tags=("number-theory", "Dickman", "rho", "enclosure", "validated"),
        examples=(
            OperationExample(
                name="through_second_interval",
                description="Enclose rho through u=2.",
                input={
                    "endpoint": {"num": "2", "den": "1"},
                    "target_width": {"mantissa": "1", "exponent": -5},
                },
            ),
        ),
    ),
)


__all__ = [
    "DICKMAN_RHO_OPERATIONS",
    "DickmanRhoAffinePiece",
    "DickmanRhoPiecewiseEnclosureRequest",
    "DickmanRhoPiecewiseEnclosureResult",
    "DyadicCoefficientBall",
    "dickman_rho_piecewise_enclosure",
]
