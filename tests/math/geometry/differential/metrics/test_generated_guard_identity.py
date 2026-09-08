"""Generated determinant identities preserve bounded retained chart loci."""

from __future__ import annotations

import pytest
from sympy import symbols

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative import (
    covariant_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy


@pytest.mark.parametrize(("scale", "offset"), ((1, 1), (2, 1), (1, 2)))
def test_generated_determinant_matches_inherited_source_guard(
    scale: int, offset: int
) -> None:
    x, _y = symbols("x y")
    axis = ("x", "y")
    inherited = canonical_locus_guards(
        tuple(
            rational_function_from_sympy(x + offset, axis).numerator
            for offset in (0, *range(2, 769))
        ),
        variable_count=2,
    )
    metric = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis)
                for value in (scale * x + offset, 1, 1, 1)
            ),
            retained_nonzero_denominators=inherited,
        )
    )
    source = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=(),
        components=(rational_function_from_sympy(1, axis),),
    )
    if offset == 2:
        with pytest.raises(OperationResourceAdmissionError, match="768 guards"):
            covariant_derivative(metric, source)
        return
    result = covariant_derivative(metric, source)
    assert all(not component.numerator.terms for component in result.components)
    assert len(result.retained_nonzero_denominators) == 768
    decoded = RationalCoordinateTensor.model_validate_json(result.model_dump_json())
    assert decoded == result
    second = covariant_derivative(metric, decoded)
    assert second.retained_nonzero_denominators == inherited
    assert all(not component.numerator.terms for component in second.components)
