"""Exact face-closed rational polytopal complexes built from maximal cells."""

from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    polytopal_complex_closure,
    spline_space,
)

__all__ = [
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "polytopal_complex_closure",
    "spline_space",
]
