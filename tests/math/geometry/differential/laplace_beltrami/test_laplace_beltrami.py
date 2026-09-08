"""Exact Laplace--Beltrami identities and retained source-locus regressions."""

from sympy import symbols

from jacobian.math.geometry.differential.laplace_beltrami import laplace_beltrami
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy

x, y = symbols("x y")


def _euclidean() -> RationalCoordinateMetric:
    axis = ("x", "y")
    components = tuple(
        rational_function_from_sympy(1 if i == j else 0, axis)
        for i in range(2)
        for j in range(2)
    )
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=components,
            retained_nonzero_denominators=(),
        )
    )


def test_harmonic_rational_scalar_retains_its_denominator_locus() -> None:
    scalar = rational_function_from_sympy(x / (x**2 + y**2), ("x", "y"))
    result = laplace_beltrami(_euclidean(), scalar)
    assert result.value.numerator.terms == ()
    expected = canonical_locus_guards(
        component_denominators=(scalar.denominator,),
        variable_count=2,
    )
    assert result.retained_nonzero_denominators == expected
