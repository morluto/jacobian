"""Exact Laplace--Beltrami identities and retained source-locus regressions."""

import pytest
from sympy import symbols

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential.laplace_beltrami import laplace_beltrami
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.values import (
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

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


def test_inherited_guards_are_counted_in_the_returned_family() -> None:
    axis = ("x", "y")
    pairs = tuple(
        sorted(((index % 16, index // 16) for index in range(256)), reverse=True)
    )
    guards = canonical_locus_guards(
        tuple(
            SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(
                            num=offset if exponents == (0, 0) else 1,
                            den=1,
                        ),
                        exponents=exponents,
                    )
                    for exponents in pairs
                )
            )
            for offset in range(2, 522)
        ),
        variable_count=2,
    )
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=_euclidean().tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    with pytest.raises(OperationResourceAdmissionError, match="allocation bounds"):
        laplace_beltrami(metric, rational_function_from_sympy(1, axis))


def test_shared_one_term_inherited_locus_remains_admitted() -> None:
    axis = ("x", "y")
    guards = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in range(1, 769)
        ),
        variable_count=2,
    )
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=_euclidean().tensor.components,
            retained_nonzero_denominators=guards,
        )
    )
    result = laplace_beltrami(metric, rational_function_from_sympy(1, axis))
    assert result.retained_nonzero_denominators == guards
