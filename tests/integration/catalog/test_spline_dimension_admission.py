"""Exact wide-coordinate spline interfaces shared by native and dispatch tests."""

from fractions import Fraction
from math import comb
from typing import Any, NoReturn

import pytest
import sympy as sp

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    PolytopalComplexClosureResult,
    SplineDimensionRequest,
    SplineDimensionResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
    spline_dimension,
)
from jacobian.math.matrices.operations import product_result
from jacobian.math.matrices.values import RationalMatrix


def wide_spline_complex() -> PolytopalComplexClosureResult:
    """Two 4-simplices on opposite sides of one dense rational 3-facet."""
    n = 10**31
    points = [
        tuple(
            Fraction(n + 17 * i + 3 * j + 1, n + 13 * i + 7 * j + 19 + j * i * i)
            for j in range(4)
        )
        for i in range(4)
    ]
    apices = [
        tuple(points[0][j] + (sign if j == 0 else 0) for j in range(4))
        for sign in (-1, 1)
    ]
    return polytopal_complex_closure(
        tuple(
            RationalVPolytope(
                space=RationalCoordinateSpace(axes=("x", "y", "z", "w")),
                vertices=tuple(
                    RationalPolytopeVertex(
                        vertex_id=f"{cell}-{index}",
                        coordinates=tuple(
                            CanonicalRational.from_fraction(v) for v in point
                        ),
                    )
                    for index, point in enumerate([*points, apex])
                ),
            )
            for cell, apex in enumerate(apices)
        )
    )


@pytest.mark.parametrize("degree,smoothness", [(4, 3), (5, 4), (3, 4), (1, -1)])
def test_wide_dimension_retains_exact_two_cell_space(
    degree: int, smoothness: int
) -> None:
    complex_value = wide_spline_complex()
    result = spline_dimension(
        SplineDimensionRequest(
            complex=complex_value, degree=degree, smoothness=smoothness
        )
    )
    monomials = comb(4 + degree, degree)
    multiples = comb(4 + degree - smoothness - 1, 4) if degree > smoothness else 0
    expected_rank = monomials - multiples if smoothness >= 0 else 0
    # The difference of the two pieces must be ell**(r+1) times an arbitrary
    # polynomial of degree <= d-r-1. Multiplication by ell**(r+1) is injective.
    assert result.rank == expected_rank
    assert result.nullity == 2 * monomials - expected_rank
    replayed = SplineDimensionResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert replayed == result
    # A global constant has the same coefficient on both cells and must be
    # annihilated by the transported compatibility matrix, unchanged.
    constant = RationalMatrix(
        row_count=2 * monomials,
        column_count=1,
        entries=tuple(
            (CanonicalRational(num=int(not any(exponents)), den=1),)
            for _, exponents in replayed.coefficient_axis
        ),
    )
    assert all(
        sum(
            entry.as_fraction() * scalar[0].as_fraction()
            for entry, scalar in zip(row, constant.entries, strict=True)
        )
        == 0
        for row in replayed.compatibility_matrix.entries
    )
    if degree < 4:
        # The general matrix-product consumer admits <=128 columns and
        # <=256-digit entries, so consume the degree-3 and empty results.
        product = product_result(replayed.compatibility_matrix, constant)
        assert all(entry.num == 0 for row in product.product.entries for entry in row)


@pytest.mark.parametrize(
    "limit,index",
    [
        ("MAX_SPLINE_DIMENSION_INTERMEDIATE_DIGITS", 0),
        ("MAX_SPLINE_DIMENSION_INTERMEDIATE_BYTES", 1),
        ("MAX_SPLINE_DIMENSION_OUTPUT_DIGITS", 2),
    ],
)
def test_dimension_refuses_source_growth_before_expansion(
    monkeypatch: pytest.MonkeyPatch, limit: str, index: int
) -> None:
    complex_value = wide_spline_complex()
    request = SplineDimensionRequest(complex=complex_value, degree=4, smoothness=3)
    _, width, row_bound, _ = spline_kernel._admit_spline_dimension(complex_value, 4, 3)
    bounds = spline_kernel._admit_spline_dimension_height(
        complex_value, 4, 3, width, row_bound
    )
    monkeypatch.setattr(spline_kernel, limit, bounds[index])
    assert spline_dimension(request).nullity == 71
    monkeypatch.setattr(spline_kernel, limit, bounds[index] - 1)

    def unexpected_expansion(*args: Any, **kwargs: Any) -> NoReturn:
        pytest.fail(
            "source refusal must precede divisor normalization/expansion and division"
        )

    catalog = Catalog.open()
    monkeypatch.setattr(sp.Poly, "monic", unexpected_expansion)
    monkeypatch.setattr(sp.Poly, "div", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError) as error:
        spline_dimension(request)
    with pytest.raises(OperationResourceAdmissionError) as dispatch_error:
        invoke_operation(
            "polyhedral_complex.spline_dimension.compute",
            request.model_dump(mode="json"),
            catalog,
        )
    assert dispatch_error.value.errors()[0]["type"] == error.value.errors()[0]["type"]
    suffix = "output" if index == 2 else "height"
    assert (
        error.value.errors()[0]["type"]
        == f"polytopal_complex.spline_dimension_{suffix}"
    )


def test_wide_spline_dimension_public_dispatch() -> None:
    complex_value = wide_spline_complex()
    result = invoke_operation(
        "polyhedral_complex.spline_dimension.compute",
        {
            "complex": complex_value.model_dump(mode="json"),
            "degree": 4,
            "smoothness": 3,
        },
        Catalog.open(),
    )
    assert result.output["rank"] == 69 and result.output["nullity"] == 71
