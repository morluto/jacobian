"""Exact face-closed rational polytopal complexes built from maximal cells."""

from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_add,
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_multiply,
    piecewise_polynomial_scalar_multiply,
    piecewise_polynomial_smoothness,
    polytopal_complex_affine_transform,
    polytopal_complex_closure,
    polytopal_complex_common_refinement,
    spline_coordinates,
    spline_dimension,
    spline_dimension_profile,
    spline_evaluate,
    spline_refinement_map,
    spline_space,
)

__all__ = [
    "piecewise_polynomial_add",
    "piecewise_polynomial_evaluate",
    "piecewise_polynomial_from_maximal_pieces",
    "piecewise_polynomial_multiply",
    "piecewise_polynomial_scalar_multiply",
    "piecewise_polynomial_smoothness",
    "polytopal_complex_affine_transform",
    "polytopal_complex_closure",
    "polytopal_complex_common_refinement",
    "spline_coordinates",
    "spline_dimension",
    "spline_dimension_profile",
    "spline_evaluate",
    "spline_refinement_map",
    "spline_space",
]
