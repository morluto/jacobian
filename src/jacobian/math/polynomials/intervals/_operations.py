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

    return PolynomialBoxEnclosureResult._from_kernel(
        request,
        enclosure=natural_interval_extension(request.polynomial, request.box),
    )


def _verify_polynomial_box_enclosure_result(
    result: PolynomialBoxEnclosureResult,
) -> bool:
    """Check an independently supplied enclosure inside the owner envelope."""

    parsed = PolynomialBoxEnclosureResult.model_validate(
        result.model_dump(mode="json"),
    )
    request = PolynomialBoxEnclosureRequest(
        polynomial=parsed.polynomial,
        box=parsed.box,
    )
    return parsed.enclosure == natural_interval_extension(
        request.polynomial,
        request.box,
    )


__all__ = ["compute_polynomial_box_enclosure"]
