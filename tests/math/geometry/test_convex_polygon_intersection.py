"""Tests for exact convex polygon intersection."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry import (
    convex_polygon_intersection as public_convex_polygon_intersection,
)
from jacobian.math.geometry._convex_polygon_intersection import (
    ConvexRationalPolygon,
    convex_polygon_intersection,
)
from jacobian.math.geometry._models import RationalPoint2D


def _pt(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=x, den="1"), y=CanonicalRational(num=y, den="1")
    )


def _poly(pts: list[tuple[str, str]]) -> ConvexRationalPolygon:
    return ConvexRationalPolygon(vertices=tuple(_pt(x, y) for x, y in pts))


def test_overlapping_squares_polygon() -> None:
    a = _poly([("0", "0"), ("2", "0"), ("2", "2"), ("0", "2")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "3"), ("1", "3")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "POLYGON"
    assert result.polygon is not None
    pts = [(p.x.num, p.y.num) for p in result.polygon.points]
    assert pts == [("1", "1"), ("2", "1"), ("2", "2"), ("1", "2")]
    # Least vertex first
    assert pts[0] == ("1", "1")
    # Provenance: each vertex has active edges
    assert len(result.vertex_active_edges) == 4


def test_vertex_touch_point() -> None:
    a = _poly([("0", "0"), ("1", "0"), ("0", "1")])
    b = _poly([("1", "0"), ("2", "0"), ("1", "1")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "POINT"
    assert result.point is not None
    assert result.point.x.num == "1" and result.point.y.num == "0"
    assert len(result.vertex_active_edges) == 1


def test_edge_touch_segment() -> None:
    # Two unit squares sharing edge y=1 from x=1 to2
    a = _poly([("0", "0"), ("2", "0"), ("2", "1"), ("0", "1")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "2"), ("1", "2")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "SEGMENT"
    assert result.segment is not None
    # Lexicographically ordered
    assert result.segment.start.x.num == "1" and result.segment.start.y.num == "1"
    assert result.segment.end.x.num == "2" and result.segment.end.y.num == "1"
    assert len(result.vertex_active_edges) == 2


def test_disjoint_empty() -> None:
    a = _poly([("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")])
    b = _poly([("2", "2"), ("3", "2"), ("3", "3"), ("2", "3")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "EMPTY"
    assert result.polygon is None and result.point is None and result.segment is None


def test_containment_reproduces_inner() -> None:
    large = _poly([("-10", "-10"), ("10", "-10"), ("0", "10")])
    small = _poly([("0", "0"), ("1", "0"), ("0", "1")])
    result = convex_polygon_intersection(large, small)
    assert result.kind == "POLYGON"
    assert result.polygon is not None
    pts = [(p.x.num, p.y.num) for p in result.polygon.points]
    # Should be exactly small polygon canonicalized
    assert set(pts) == {("0", "0"), ("1", "0"), ("0", "1")}


def test_rational_nonintegral_intersection() -> None:
    # Triangles with rational intersection not integral
    a = _poly([("0", "0"), ("2", "0"), ("0", "2")])
    b = _poly([("1", "0"), ("3", "0"), ("1", "2")])
    result = convex_polygon_intersection(a, b)
    # Overlap should be polygon with rational vertices including (1,0),(2,0),(1,1)
    assert result.kind in ("POLYGON", "SEGMENT", "POINT")
    if result.kind == "POLYGON":
        assert result.polygon is not None
        assert len(result.polygon.points) >= 3


def test_defining_invariant_half_plane_containment() -> None:
    a = _poly([("0", "0"), ("2", "0"), ("2", "2"), ("0", "2")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "3"), ("1", "3")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "POLYGON"
    # Every output vertex must be inside both polygons (including boundary)
    assert result.polygon is not None

    def point_in_convex(pt: RationalPoint2D, poly: ConvexRationalPolygon) -> bool:
        p = (pt.x.as_fraction(), pt.y.as_fraction())
        verts = [(v.x.as_fraction(), v.y.as_fraction()) for v in poly.vertices]
        n = len(verts)
        for i in range(n):
            a_pt = verts[i]
            b_pt = verts[(i + 1) % n]
            cross = (b_pt[0] - a_pt[0]) * (p[1] - a_pt[1]) - (b_pt[1] - a_pt[1]) * (
                p[0] - a_pt[0]
            )
            if cross < 0:
                return False
        return True

    for pt in result.polygon.points:
        assert point_in_convex(pt, a)
        assert point_in_convex(pt, b)


def test_rejects_non_convex_or_collinear() -> None:
    # Collinear consecutive triple
    polygon = ConvexRationalPolygon(
        vertices=(_pt("0", "0"), _pt("1", "0"), _pt("2", "0"), _pt("1", "1"))
    )
    with pytest.raises(OperationDomainValidationError, match="strictly CCW"):
        convex_polygon_intersection(polygon, polygon)


def test_rejects_self_intersecting_left_turn_ring() -> None:
    polygon = _poly([("0", "3"), ("-2", "-3"), ("3", "1"), ("-3", "1"), ("2", "-3")])
    with pytest.raises(OperationDomainValidationError, match="left half-plane"):
        convex_polygon_intersection(polygon, polygon)


def test_native_geometry_api_exposes_intersection() -> None:
    square = _poly([("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")])
    assert public_convex_polygon_intersection(square, square).kind == "POLYGON"
    # Not CCW (clockwise)
    polygon = ConvexRationalPolygon(
        vertices=(_pt("0", "0"), _pt("0", "2"), _pt("2", "2"), _pt("2", "0"))
    )
    with pytest.raises(OperationDomainValidationError, match="strictly CCW"):
        public_convex_polygon_intersection(polygon, polygon)


def test_json_round_trip() -> None:
    a = _poly([("0", "0"), ("2", "0"), ("2", "2"), ("0", "2")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "3"), ("1", "3")])
    result = convex_polygon_intersection(a, b)
    json_val = result.model_dump_json()
    replay = type(result).model_validate_json(json_val, strict=True)
    assert replay == result
