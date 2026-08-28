"""Exact rational-polynomial interval operations."""

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.polynomials.intervals._kernel import natural_interval_extension
from jacobian.math.polynomials.intervals._models import _require_enclosure_preflight
from jacobian.math.polynomials.values import RationalPolynomial


def polynomial_box_enclosure(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> ClosedRationalInterval:
    """Return the admitted natural interval extension on a complete box."""

    try:
        _require_enclosure_preflight(polynomial, box)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("polynomial", "box"), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("polynomial", "box"),
            code="polynomial.box_admission",
            message=str(exc),
        ) from exc
    return natural_interval_extension(polynomial, box)


__all__ = ["polynomial_box_enclosure"]
