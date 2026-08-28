"""Observed-work evidence for the V-polytope extremality budget."""

from __future__ import annotations

from math import comb
from typing import Any
from unittest.mock import patch

from sympy.matrices.dense import MutableDenseMatrix
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.operations import (
    require_full_dimensional_extreme_vertices,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=str(value), den="1")


def _square() -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            RationalPolytopeVertex(
                vertex_id="a", coordinates=(_rational(0), _rational(0))
            ),
            RationalPolytopeVertex(
                vertex_id="b", coordinates=(_rational(1), _rational(0))
            ),
            RationalPolytopeVertex(
                vertex_id="c", coordinates=(_rational(1), _rational(1))
            ),
            RationalPolytopeVertex(
                vertex_id="d", coordinates=(_rational(0), _rational(1))
            ),
        ),
    )


def test_extremality_charges_every_observed_orientation_determinant() -> None:
    polytope = _square()
    dimension = len(polytope.space.axes)
    vertex_count = len(polytope.vertices)
    charged_orientation_tests = (
        comb(vertex_count, dimension) * (vertex_count - dimension) + vertex_count + 1
    )

    original_determinant = MutableDenseMatrix.det
    executions = {"orientation_determinant": 0}

    def counted_determinant(*args: Any, **kwargs: Any) -> Any:
        executions["orientation_determinant"] += 1
        return original_determinant(*args, **kwargs)

    with patch.object(
        MutableDenseMatrix, "det", autospec=True, side_effect=counted_determinant
    ):
        require_full_dimensional_extreme_vertices(polytope)

    assert_charged_work_parity(
        charged={"orientation_determinant": charged_orientation_tests},
        executed=executions,
    )
