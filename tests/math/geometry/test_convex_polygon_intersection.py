"""Tests for exact convex polygon intersection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
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


def test_catalog_contains_convex_intersection():
    from jacobian.catalog.catalog import Catalog

    cat = Catalog.open()
    assert cat.inspect("geometry.convex_polygon.intersection.compute") is not None


def test_overlapping_squares_polygon():
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


def test_vertex_touch_point():
    a = _poly([("0", "0"), ("1", "0"), ("0", "1")])
    b = _poly([("1", "0"), ("2", "0"), ("1", "1")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "POINT"
    assert result.point is not None
    assert result.point.x.num == "1" and result.point.y.num == "0"
    assert len(result.vertex_active_edges) == 1


def test_edge_touch_segment():
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


def test_disjoint_empty():
    a = _poly([("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")])
    b = _poly([("2", "2"), ("3", "2"), ("3", "3"), ("2", "3")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "EMPTY"
    assert result.polygon is None and result.point is None and result.segment is None


def test_containment_reproduces_inner():
    large = _poly([("-10", "-10"), ("10", "-10"), ("0", "10")])
    small = _poly([("0", "0"), ("1", "0"), ("0", "1")])
    result = convex_polygon_intersection(large, small)
    assert result.kind == "POLYGON"
    assert result.polygon is not None
    pts = [(p.x.num, p.y.num) for p in result.polygon.points]
    # Should be exactly small polygon canonicalized
    assert set(pts) == {("0", "0"), ("1", "0"), ("0", "1")}


def test_rational_nonintegral_intersection():
    # Triangles with rational intersection not integral
    a = _poly([("0", "0"), ("2", "0"), ("0", "2")])
    b = _poly([("1", "0"), ("3", "0"), ("1", "2")])
    result = convex_polygon_intersection(a, b)
    # Overlap should be polygon with rational vertices including (1,0),(2,0),(1,1)
    assert result.kind in ("POLYGON", "SEGMENT", "POINT")
    if result.kind == "POLYGON":
        assert len(result.polygon.points) >= 3


def test_defining_invariant_half_plane_containment():
    a = _poly([("0", "0"), ("2", "0"), ("2", "2"), ("0", "2")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "3"), ("1", "3")])
    result = convex_polygon_intersection(a, b)
    assert result.kind == "POLYGON"
    # Every output vertex must be inside both polygons (including boundary)

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


def test_rejects_non_convex_or_collinear():
    # Collinear consecutive triple
    with pytest.raises(ValidationError, match=r"strict.*CCW|collinear"):
        ConvexRationalPolygon(
            vertices=(_pt("0", "0"), _pt("1", "0"), _pt("2", "0"), _pt("1", "1"))
        )
    # Not CCW (clockwise)
    with pytest.raises(ValidationError, match=r"strict.*CCW"):
        ConvexRationalPolygon(
            vertices=(_pt("0", "0"), _pt("0", "2"), _pt("2", "2"), _pt("2", "0"))
        )


def test_via_catalog_example_replay():
    from jacobian.catalog.catalog import Catalog

    cat = Catalog.open()
    binding = cat._binding("geometry.convex_polygon.intersection.compute")
    req = binding.request_type.model_validate(
        {
            "polygon_a": {
                "vertices": [
                    {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                    {"x": {"num": "2", "den": "1"}, "y": {"num": "0", "den": "1"}},
                    {"x": {"num": "2", "den": "1"}, "y": {"num": "2", "den": "1"}},
                    {"x": {"num": "0", "den": "1"}, "y": {"num": "2", "den": "1"}},
                ]
            },
            "polygon_b": {
                "vertices": [
                    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    {"x": {"num": "3", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    {"x": {"num": "3", "den": "1"}, "y": {"num": "3", "den": "1"}},
                    {"x": {"num": "1", "den": "1"}, "y": {"num": "3", "den": "1"}},
                ]
            },
        }
    )
    res = binding.run(req)
    assert res.kind == "POLYGON"


def test_json_round_trip():
    a = _poly([("0", "0"), ("2", "0"), ("2", "2"), ("0", "2")])
    b = _poly([("1", "1"), ("3", "1"), ("3", "3"), ("1", "3")])
    result = convex_polygon_intersection(a, b)
    json_val = result.model_dump_json()
    replay = type(result).model_validate_json(json_val, strict=True)
    assert replay == result
