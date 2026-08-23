"""Wire adapters for circle inversion operations."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry.inversion._models import (
    CircleInversionRequest,
    CircleInversionResult,
)


def compute_circle_inversion(request: CircleInversionRequest) -> CircleInversionResult:
    """Compute exact circle inversion of a rational planar point."""
    image_x, image_y = invert_point(
        request.center.x.as_fraction(),
        request.center.y.as_fraction(),
        request.power.as_fraction(),
        request.point.x.as_fraction(),
        request.point.y.as_fraction(),
    )
    return CircleInversionResult(
        center=request.center,
        power=request.power,
        point=request.point,
        image=RationalPoint2D(
            x=CanonicalRational.from_fraction(image_x),
            y=CanonicalRational.from_fraction(image_y),
        ),
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
