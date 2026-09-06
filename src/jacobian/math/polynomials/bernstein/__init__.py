"""Exact tensor-product Bernstein representations on rational boxes."""

from jacobian.math.polynomials.bernstein.operations import (
    bernstein_coefficients,
    restrict_bernstein,
    verify_bernstein_coefficients,
    verify_bernstein_restriction,
)
from jacobian.math.polynomials.bernstein.values import RationalBernsteinPolynomial

__all__ = [
    "RationalBernsteinPolynomial",
    "bernstein_coefficients",
    "restrict_bernstein",
    "verify_bernstein_coefficients",
    "verify_bernstein_restriction",
]
