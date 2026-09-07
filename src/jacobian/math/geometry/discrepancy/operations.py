"""Exact unnormalized squared L2 star discrepancy on a rational unit cube."""

from fractions import Fraction
from math import lcm, prod

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import RationalMatrix


def squared_l2_star_discrepancy(points: RationalMatrix) -> CanonicalRational:
    """Integrate (count([0,u)) - n*prod(u))^2 against Lebesgue measure.

    Rows are point occurrences, columns are ordered coordinates; repeated and
    boundary points remain distinct occurrences. The empty list has value 0.
    Expanding the square and integrating each term gives Warnock's formula:
    sum_ij prod(1-max(x_i,x_j)) - n/2^(d-1) sum_i prod(1-x_i^2) + n^2/3^d.
    """
    request_checkpoint("before discrepancy admission")
    n, d = points.row_count, points.column_count
    if d < 1:
        raise OperationDomainValidationError(
            location=("points",),
            code="geometry.discrepancy_dimension",
            message="points must retain a positive ambient dimension in column_count",
        )
    if not n:
        return CanonicalRational(num=0, den=1)
    work = (n * n + n) * d
    if work > 1_000_000:
        raise OperationDomainValidationError(
            location=("points",),
            code="geometry.discrepancy_work",
            message="discrepancy requires (n*n+n)*d <= 1000000",
        )
    denominator = 1
    for row in points.entries:
        for coordinate in row:
            if not 0 <= coordinate.num <= coordinate.den:
                raise OperationDomainValidationError(
                    location=("points",),
                    code="geometry.discrepancy_unit_cube",
                    message="every point coordinate must lie in [0,1]",
                )
            denominator = lcm(denominator, coordinate.den)
            if denominator.bit_length() > 24_000:
                raise OperationDomainValidationError(
                    location=("points",),
                    code="geometry.discrepancy_growth",
                    message="coordinate common denominator exceeds the working height envelope",
                )
    # A common denominator is D^(2d)*2^(d-1)*3^d. Every pair term is
    # in [0,1], so 4*n^2 times this denominator bounds all intermediates.
    bits = 2 * d * denominator.bit_length() + 3 * d + 2 * n.bit_length() + 3
    if bits > 96_000 or work * max(1, (bits + 63) // 64) ** 2 > 50_000_000:
        raise OperationDomainValidationError(
            location=("points",),
            code="geometry.discrepancy_growth",
            message="discrepancy exceeds 50000000 weighted arithmetic units or the exact scalar height envelope",
        )
    coordinates = [
        [v.num * (denominator // v.den) for v in row] for row in points.entries
    ]
    pairs = sum(
        prod(denominator - max(x, y) for x, y in zip(a, b, strict=True))
        for a in coordinates
        for b in coordinates
    )
    linear = sum(prod(denominator**2 - x * x for x in row) for row in coordinates)
    value = (
        Fraction(pairs, denominator**d)
        - Fraction(n * linear, 2 ** (d - 1) * denominator ** (2 * d))
        + Fraction(n * n, 3**d)
    )
    request_checkpoint("before discrepancy result construction")
    return CanonicalRational.from_fraction(value)


__all__ = ["squared_l2_star_discrepancy"]
