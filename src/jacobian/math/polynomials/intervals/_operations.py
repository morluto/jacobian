"""Domain operation for exact rational-polynomial box enclosure."""

from jacobian.math.polynomials.intervals._kernel import natural_interval_extension
from jacobian.math.polynomials.intervals._models import (
    PolynomialBoxEnclosureRequest,
    PolynomialBoxEnclosureResult,
)


def compute_polynomial_box_enclosure(
    request: PolynomialBoxEnclosureRequest,
) -> PolynomialBoxEnclosureResult:
    """Return the deterministic natural interval extension on the complete box."""

    return PolynomialBoxEnclosureResult(
        polynomial=request.polynomial,
        box=request.box,
        enclosure=natural_interval_extension(request.polynomial, request.box),
    )


__all__ = ["compute_polynomial_box_enclosure"]
