"""Wire adapters for circle inversion operations."""

from jacobian._exact import CanonicalRational
from fractions import Fraction
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)


def compute_circle_inversion(request: CircleInversionRequest) -> CircleInversionResult:
    """Compute exact circle inversion of a rational planar point."""

    cx = request.center_x.as_fraction()
    cy = request.center_y.as_fraction()
    s = request.power.as_fraction()
    px = request.point_x.as_fraction()
    py = request.point_y.as_fraction()

    result = invert_point(cx, cy, s, px, py)
    return CircleInversionResult(
        center_x=request.center_x,
        center_y=request.center_y,
        power=request.power,
        point_x=request.point_x,
        point_y=request.point_y,
        inverted_x=CanonicalRational.from_fraction(result[0]),
        inverted_y=CanonicalRational.from_fraction(result[1]),
    )


def invert_point(
    cx: Fraction,
    cy: Fraction,
    s: Fraction,
    px: Fraction,
    py: Fraction,
) -> tuple[Fraction, Fraction]:
    """Compute exact circle inversion I_{c,s}(p).

    Given center c=(cx,cy), positive rational inversion power s
    (squared inversion radius), and point p=(px,py) ≠ c, returns
    q = c + (s / ||p-c||²) * (p-c).

    Raises ValueError if p = c.
    """

    dx = px - cx
    dy = py - cy
    norm_sq = dx * dx + dy * dy
    if norm_sq == 0:
        raise ValueError("the inversion center cannot be inverted")
    scale = s / norm_sq
    qx = cx + scale * dx
    qy = cy + scale * dy
    return qx, qy


__all__ = ["compute_circle_inversion", "invert_point"]
