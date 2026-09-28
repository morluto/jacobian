"""Typed request for exact sparse free-algebra polynomial subtraction."""

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial


class FreeAlgebraPolynomialSubtractionRequest(StrictModel):
    """Subtract polynomials over the same alphabet.

    Each operand admits at most 128 terms; the union of their supports must
    contain at most 128 distinct words. Overlapping terms count once toward
    the result support bound.
    """

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial


__all__ = ["FreeAlgebraPolynomialSubtractionRequest"]
