"""Typed request for exact sparse free-algebra polynomial subtraction."""

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial


class FreeAlgebraPolynomialSubtractionRequest(StrictModel):
    """Subtract two canonical polynomials over the same ordered alphabet."""

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial


__all__ = ["FreeAlgebraPolynomialSubtractionRequest"]
