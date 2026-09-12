"""Exact Ehrhart interpolation for integral polytopes (#1192)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.geometry.polytopes.lattice._models import EhrhartRequest
from jacobian.math.geometry.polytopes.lattice._tools import ehrhart_polynomial


def _vertex(*coordinates: int) -> dict[str, list[dict[str, int]]]:
    return {"coordinates": [{"num": value, "den": 1} for value in coordinates]}


def test_unit_square_ehrhart_polynomial_and_dilation_counts() -> None:
    request = EhrhartRequest(
        vertices=[_vertex(0, 0), _vertex(1, 0), _vertex(0, 1), _vertex(1, 1)],
        degree_bound=2,
        max_dilation=4,
    )
    result = ehrhart_polynomial(request)
    assert result.counts == ((0, 1), (1, 4), (2, 9), (3, 16), (4, 25))
    assert [(item.num, item.den) for item in result.coefficients] == [
        (1, 1),
        (2, 1),
        (1, 1),
    ]


def test_unit_interval_has_zero_dilate_and_exact_linear_coefficients() -> None:
    result = ehrhart_polynomial(
        EhrhartRequest(
            vertices=[_vertex(0), _vertex(1)], degree_bound=1, max_dilation=3
        )
    )
    assert result.counts == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert [(item.num, item.den) for item in result.coefficients] == [(1, 1), (1, 1)]


def test_rational_vertices_are_rejected_until_quasipolynomial_scope_exists() -> None:
    with pytest.raises(ValidationError, match="requires integral vertices"):
        EhrhartRequest(
            vertices=[
                {"coordinates": [{"num": 0, "den": 1}]},
                {"coordinates": [{"num": 1, "den": 2}]},
            ],
            degree_bound=1,
        )


def test_degree_bound_must_cover_dimension() -> None:
    with pytest.raises(ValidationError, match="cover the polytope dimension"):
        EhrhartRequest(
            vertices=[_vertex(0, 0), _vertex(1, 0), _vertex(0, 1)], degree_bound=1
        )
