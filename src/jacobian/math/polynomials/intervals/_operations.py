"""Domain operation for exact rational-polynomial box enclosure."""

from jacobian.math.polynomials.intervals._kernel import natural_interval_extension
from jacobian.math.polynomials.intervals._models import (
    PolynomialBoxEnclosureRequest,
    PolynomialBoxEnclosureResult,
    polynomial_box_source_digest,
)


def compute_polynomial_box_enclosure(
    request: PolynomialBoxEnclosureRequest,
) -> PolynomialBoxEnclosureResult:
    """Return the deterministic natural interval extension on the complete box."""

    # The parsed request already ran the complete admission preflight, and each
    # value below is produced by the typed kernel. Independently supplied wire
    # results still replay through PolynomialBoxEnclosureResult.model_validate.
    return PolynomialBoxEnclosureResult.model_construct(
        polynomial=request.polynomial,
        box=request.box,
        source_digest=polynomial_box_source_digest(request.polynomial, request.box),
        enclosure=natural_interval_extension(request.polynomial, request.box),
    )


__all__ = ["compute_polynomial_box_enclosure"]
