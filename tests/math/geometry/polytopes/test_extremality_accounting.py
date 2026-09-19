"""Observed-work evidence for the V-polytope extremality budget."""

from __future__ import annotations

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
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    dd_work_bound,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    points_to_facets as convert_points_to_facets,
)
from jacobian.math.geometry.polytopes.operations import (
    require_full_dimensional_extreme_vertices,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


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


def test_extremality_charges_dd_pairs_and_rank_tests() -> None:
    polytope = _square()
    dimension = len(polytope.space.axes)
    vertex_count = len(polytope.vertices)
    _ray_bound, charged_candidate_pairs = dd_work_bound(
        vertex_count, dimension + 1
    )
    charged_rank_tests = vertex_count + 1

    original_rank = MutableDenseMatrix.rank
    executions = {"candidate_pair": 0, "rank_test": 0}

    def counted_rank(*args: Any, **kwargs: Any) -> Any:
        executions["rank_test"] += 1
        return original_rank(*args, **kwargs)

    def counted_conversion(*args: Any, **kwargs: Any) -> Any:
        conversion = convert_points_to_facets(*args, **kwargs)
        executions["candidate_pair"] += conversion.cone.candidate_pairs
        return conversion

    with (
        patch.object(MutableDenseMatrix, "rank", autospec=True, side_effect=counted_rank),
        patch(
            "jacobian.math.geometry.polytopes.operations.points_to_facets",
            side_effect=counted_conversion,
        ),
    ):
        require_full_dimensional_extreme_vertices(polytope)

    assert_charged_work_parity(
        charged={
            "candidate_pair": charged_candidate_pairs,
            "rank_test": charged_rank_tests,
        },
        executed=executions,
    )
