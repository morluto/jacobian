"""Exact order in one real quadratic field."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.exact import CanonicalRational, require_bounded_rational
from jacobian.contracts.results import ContractModel

_MAX_RADICAND = 1_000_000
_MAX_DIGITS = 256
RealQuadraticSignBasis = Literal[
    "RATIONAL_ONLY",
    "RADICAL_ONLY",
    "SAME_SIGN",
    "OPPOSING_SIGNS_SQUARED_MAGNITUDES",
]


def _is_square_free(value: int) -> bool:
    return all(value % (divisor * divisor) for divisor in range(2, isqrt(value) + 1))


def _order(left: Fraction, right: Fraction) -> Literal["LT", "EQ", "GT"]:
    return "LT" if left < right else "GT" if left > right else "EQ"


def _sign(a: Fraction, b: Fraction, d: int) -> int:
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    rational_square = a * a
    radical_square = b * b * d
    if rational_square == radical_square:
        raise ValueError("square-free quadratic magnitudes cannot tie")
    dominant = b if radical_square > rational_square else a
    return (dominant > 0) - (dominant < 0)


class RealQuadraticValue(ContractModel):
    rational_part: CanonicalRational
    radical_coefficient: CanonicalRational
    radicand: StrictInt = Field(ge=2, le=_MAX_RADICAND)

    @model_validator(mode="after")
    def require_canonical_field_value(self) -> Self:
        require_bounded_rational(
            self.rational_part, max_digits=_MAX_DIGITS, label="rational part"
        )
        require_bounded_rational(
            self.radical_coefficient,
            max_digits=_MAX_DIGITS,
            label="radical coefficient",
        )
        if not _is_square_free(self.radicand):
            raise ValueError("real-quadratic radicand must be square-free")
        return self


class RealQuadraticOrderRequest(ContractModel):
    left: RealQuadraticValue
    right: RealQuadraticValue

    @model_validator(mode="after")
    def require_shared_field(self) -> Self:
        if self.left.radicand != self.right.radicand:
            raise ValueError("comparison requires one shared radicand")
        return self


class RealQuadraticSignCertificate(ContractModel):
    rational_part_squared: CanonicalRational
    radical_part_squared: CanonicalRational
    magnitude_order: Literal["LT", "EQ", "GT"]


class RealQuadraticOrderValue(ContractModel):
    left: RealQuadraticValue
    right: RealQuadraticValue
    difference: RealQuadraticValue
    order: Literal["LT", "EQ", "GT"]
    sign_basis: RealQuadraticSignBasis
    sign_certificate: RealQuadraticSignCertificate

    @model_validator(mode="after")
    def bind_exact_order(self) -> Self:
        a = (
            self.left.rational_part.as_fraction()
            - self.right.rational_part.as_fraction()
        )
        b = (
            self.left.radical_coefficient.as_fraction()
            - self.right.radical_coefficient.as_fraction()
        )
        if (
            self.difference.radicand != self.left.radicand
            or self.difference.rational_part.as_fraction() != a
            or self.difference.radical_coefficient.as_fraction() != b
        ):
            raise ValueError("difference must equal left minus right")
        expected_order = (
            "LT"
            if _sign(a, b, self.left.radicand) < 0
            else "GT"
            if _sign(a, b, self.left.radicand) > 0
            else "EQ"
        )
        if self.order != expected_order:
            raise ValueError("order must match exact quadratic sign")
        expected_basis: RealQuadraticSignBasis = (
            "RATIONAL_ONLY"
            if b == 0
            else "RADICAL_ONLY"
            if a == 0
            else "SAME_SIGN"
            if (a > 0) == (b > 0)
            else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
        )
        if self.sign_basis != expected_basis:
            raise ValueError("sign basis does not match difference structure")
        rational_square = a * a
        radical_square = b * b * self.left.radicand
        if (
            self.sign_certificate.rational_part_squared.as_fraction() != rational_square
            or self.sign_certificate.radical_part_squared.as_fraction()
            != radical_square
            or self.sign_certificate.magnitude_order
            != _order(rational_square, radical_square)
        ):
            raise ValueError("sign certificate does not match squared magnitudes")
        return self


def real_quadratic_order(
    request: RealQuadraticOrderRequest,
) -> RealQuadraticOrderValue:
    a = (
        request.left.rational_part.as_fraction()
        - request.right.rational_part.as_fraction()
    )
    b = (
        request.left.radical_coefficient.as_fraction()
        - request.right.radical_coefficient.as_fraction()
    )
    d = request.left.radicand
    sign = _sign(a, b, d)
    basis: RealQuadraticSignBasis = (
        "RATIONAL_ONLY"
        if b == 0
        else "RADICAL_ONLY"
        if a == 0
        else "SAME_SIGN"
        if (a > 0) == (b > 0)
        else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
    )
    rational_square = a * a
    radical_square = b * b * d
    return RealQuadraticOrderValue(
        left=request.left,
        right=request.right,
        difference=RealQuadraticValue(
            rational_part=CanonicalRational.from_fraction(a),
            radical_coefficient=CanonicalRational.from_fraction(b),
            radicand=d,
        ),
        order="LT" if sign < 0 else "GT" if sign > 0 else "EQ",
        sign_basis=basis,
        sign_certificate=RealQuadraticSignCertificate(
            rational_part_squared=CanonicalRational.from_fraction(rational_square),
            radical_part_squared=CanonicalRational.from_fraction(radical_square),
            magnitude_order=_order(rational_square, radical_square),
        ),
    )


__all__ = [
    "RealQuadraticOrderRequest",
    "RealQuadraticOrderValue",
    "RealQuadraticSignCertificate",
    "RealQuadraticValue",
    "real_quadratic_order",
]
