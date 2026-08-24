"""Tests for exact rational V-polytope support and exposed faces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polytope import (
    PolytopeSupportResult,
    RationalCoordinateSpace,
    RationalCovector,
    RationalExposedFace,
    RationalPolytopeVertex,
    RationalVPolytope,
    polytope_support,
)
from jacobian.math.polytope._models import (
    MAX_SUPPORT_COMPONENT_DIGITS,
    PolytopeSupportRequest,
)
from jacobian.math.polytope._operations import compute_polytope_support


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=str(value), den="1")


def _vertex(vertex_id: str, *coordinates: int) -> RationalPolytopeVertex:
    return RationalPolytopeVertex(
        vertex_id=vertex_id,
        coordinates=tuple(_rational(value) for value in coordinates),
    )


def _square() -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x", "y")),
        vertices=(
            _vertex("bottom_left", 0, 0),
            _vertex("bottom_right", 1, 0),
            _vertex("top_left", 0, 1),
            _vertex("top_right", 1, 1),
        ),
    )


def _covector(*components: int) -> RationalCovector:
    return RationalCovector(
        space=RationalCoordinateSpace(axes=("x", "y")),
        components=tuple(_rational(component) for component in components),
    )


def test_support_returns_exact_value_and_complete_exposed_edge() -> None:
    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )

    assert result.support_value.as_fraction() == 1
    assert tuple(vertex.vertex_id for vertex in result.exposed_face.vertices) == (
        "top_left",
        "top_right",
    )
    assert result.exposed_face.space == result.polytope.space


def test_zero_covector_exposes_the_whole_polytope() -> None:
    polytope = _square()
    result = polytope_support(polytope, _covector(0, 0))

    assert result.support_value.as_fraction() == 0
    assert result.exposed_face.vertices == polytope.vertices


def test_positive_covector_scaling_preserves_face_and_scales_support() -> None:
    polytope = _square()
    unit = polytope_support(polytope, _covector(0, 1))
    scaled = polytope_support(polytope, _covector(0, 3))

    assert scaled.support_value.as_fraction() == 3 * unit.support_value.as_fraction()
    assert scaled.exposed_face == unit.exposed_face


def test_support_is_source_bound_against_forged_value_or_face() -> None:
    result = compute_polytope_support(
        PolytopeSupportRequest(polytope=_square(), covector=_covector(0, 1))
    )

    with pytest.raises(ValidationError, match="support value must equal"):
        PolytopeSupportResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "support_value": {"num": "2", "den": "1"},
            }
        )
    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult(
            polytope=result.polytope,
            covector=result.covector,
            support_value=result.support_value,
            exposed_face=RationalExposedFace(
                space=result.polytope.space,
                vertices=(result.polytope.vertices[0],),
            ),
        )
    with pytest.raises(ValidationError, match="exposed face must be exactly"):
        PolytopeSupportResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "covector": {
                    "space": {"axes": ["x", "y"]},
                    "components": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
            }
        )


def test_request_rejects_coordinate_axis_mismatch() -> None:
    bad_covector = RationalCovector(
        space=RationalCoordinateSpace(axes=("y", "x")),
        components=(_rational(0), _rational(1)),
    )

    with pytest.raises(ValidationError, match="same coordinate space"):
        PolytopeSupportRequest(polytope=_square(), covector=bad_covector)


def test_v_polytope_rejects_nonextreme_vertex() -> None:
    with pytest.raises(ValidationError, match="exact extreme vertices"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=(
                _vertex("bottom_left", 0, 0),
                _vertex("bottom_right", 2, 0),
                _vertex("middle", 1, 0),
                _vertex("top_left", 0, 2),
                _vertex("top_right", 2, 2),
            ),
        )


def test_v_polytope_rejects_lower_dimensional_hull() -> None:
    with pytest.raises(ValidationError, match="affinely span"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=(
                _vertex("left", 0, 0),
                _vertex("middle", 1, 0),
                _vertex("right", 2, 0),
            ),
        )


def test_support_component_bound_precedes_hull_expansion() -> None:
    over_bound = CanonicalRational(
        num="1" + "0" * MAX_SUPPORT_COMPONENT_DIGITS, den="1"
    )
    with pytest.raises(ValidationError, match="String should match pattern"):
        RationalCovector(
            space=RationalCoordinateSpace(axes=("x", "y")),
            components=(over_bound, _rational(0)),
        )


def test_over_bound_vertex_coordinate_is_rejected_before_hull_predicates() -> None:
    over_bound = CanonicalRational(
        num="1" + "0" * MAX_SUPPORT_COMPONENT_DIGITS, den="1"
    )
    with pytest.raises(ValidationError, match="String should match pattern"):
        RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=(
                RationalPolytopeVertex(
                    vertex_id="a",
                    coordinates=(_rational(0), _rational(0)),
                ),
                RationalPolytopeVertex(
                    vertex_id="b",
                    coordinates=(over_bound, _rational(0)),
                ),
                RationalPolytopeVertex(
                    vertex_id="c",
                    coordinates=(_rational(0), _rational(1)),
                ),
            ),
        )


def test_extremality_subfacet_bound_is_enforced_before_filtering() -> None:
    axes = ("a", "b", "c", "d", "e")
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=tuple(_rational(index ** (power + 1)) for power in range(5)),
        )
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="subfacet bound"):
        RationalVPolytope(space=RationalCoordinateSpace(axes=axes), vertices=vertices)


def test_extremality_orientation_work_is_enforced_before_filtering() -> None:
    axes = ("x", "y", "z")
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:02d}",
            coordinates=(_rational(index), _rational(index**2), _rational(index**3)),
        )
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="orientation-test bound"):
        RationalVPolytope(space=RationalCoordinateSpace(axes=axes), vertices=vertices)


def test_support_schema_publishes_component_and_hull_work_bounds() -> None:
    schema = PolytopeSupportRequest.model_json_schema()
    bounded_rational = schema["$defs"]["SupportBoundedRational"]["properties"]
    vertex_description = schema["$defs"]["RationalVPolytope"]["properties"]["vertices"][
        "description"
    ]

    assert bounded_rational["num"]["maxLength"] == MAX_SUPPORT_COMPONENT_DIGITS + 1
    assert bounded_rational["den"]["maxLength"] == MAX_SUPPORT_COMPONENT_DIGITS
    assert "C(n,d)" in vertex_description
    assert "orientation tests" in vertex_description
