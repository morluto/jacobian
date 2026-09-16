"""Tests for the exact rational-polytope edge-profile and vertex-figure operations."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
    polytope_edge_profile,
    polytope_vertex_figure,
)
from jacobian.math.geometry.polytopes._models import (
    MAX_FACET_SIGN_TESTS,
    EdgeProfileRequest,
    VertexFigureRequest,
)
from jacobian.math.geometry.polytopes._rational_geometry import facets_from_points
from jacobian.math.geometry.polytopes._tools import (
    TOOLS,
    compute_polytope_edge_profile,
    compute_polytope_vertex_figure,
)
from jacobian.math.geometry.polytopes.operations import convex_hull_volume


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _vertex(vertex_id: str, *coordinates: int) -> RationalPolytopeVertex:
    return RationalPolytopeVertex(
        vertex_id=vertex_id,
        coordinates=tuple(_rational(value) for value in coordinates),
    )


def _polytope(
    axes: tuple[str, ...], rows: tuple[tuple[str, tuple[int, ...]], ...]
) -> RationalVPolytope:
    ordered = tuple(sorted(rows, key=lambda row: row[0]))
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=axes),
        vertices=tuple(_vertex(vid, *coords) for vid, coords in ordered),
    )


def _segment(ids: tuple[str, str] = ("left", "right")) -> RationalVPolytope:
    return _polytope(("x",), ((ids[0], (0,)), (ids[1], (1,))))


def _square() -> RationalVPolytope:
    return _polytope(
        ("x", "y"),
        (
            ("bottom_left", (0, 0)),
            ("bottom_right", (1, 0)),
            ("top_left", (0, 1)),
            ("top_right", (1, 1)),
        ),
    )


def _cube() -> RationalVPolytope:
    return _polytope(
        ("x", "y", "z"),
        tuple(
            (f"v{i}{j}{k}", (i, j, k)) for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ),
    )


def _tetrahedron() -> RationalVPolytope:
    return _polytope(
        ("x", "y", "z"),
        (
            ("origin", (0, 0, 0)),
            ("ex", (1, 0, 0)),
            ("ey", (0, 1, 0)),
            ("ez", (0, 0, 1)),
        ),
    )


def _coords(polytope: RationalVPolytope) -> dict[str, tuple[Fraction, ...]]:
    return {
        vertex.vertex_id: tuple(c.as_fraction() for c in vertex.coordinates)
        for vertex in polytope.vertices
    }


def _edge_set(polytope: RationalVPolytope) -> set[tuple[str, str]]:
    result = polytope_edge_profile(polytope)
    return {(edge.endpoint_a, edge.endpoint_b) for edge in result.edges}


def _raw_points(polytope: RationalVPolytope) -> list[list[Rational]]:
    return [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in polytope.vertices
    ]


def _oracle_is_edge(
    points: list[list[Rational]], dim: int, first: int, second: int
) -> bool:
    """Independent brute-force edge test from the supporting-hyperplane definition.

    A pair is an edge iff some hyperplane through a d-subset containing the
    pair supports the point set one-sidedly and the pair is adjacent within
    that hyperplane's restricted point family (recursion bottoms out at
    dim 1 as strict betweenness). This searches candidate hyperplanes
    existentially instead of reusing the kernel's global facet profile and
    incidence-intersection rank rule.
    """

    if points[first] == points[second]:
        return False
    if dim == 1:
        line = sorted({point[0] for point in points})
        low, high = points[first][0], points[second][0]
        if low > high:
            low, high = high, low
        return not any(low < value < high for value in line)
    for subset in combinations(range(len(points)), dim):
        if first not in subset or second not in subset:
            continue
        base = [points[index] for index in subset]
        spans = Matrix(
            [
                [base[t][axis] - base[0][axis] for axis in range(dim)]
                for t in range(1, dim)
            ]
        )
        nullspace = spans.nullspace()
        if len(nullspace) != 1:
            continue
        normal = nullspace[0]
        offset = sum(normal[axis] * base[0][axis] for axis in range(dim))
        residuals = [
            sum(normal[axis] * point[axis] for axis in range(dim)) - offset
            for point in points
        ]
        if not (
            all(value <= 0 for value in residuals)
            or all(value >= 0 for value in residuals)
        ):
            continue
        on_plane = [index for index in range(len(points)) if residuals[index] == 0]
        if len(on_plane) <= 2:
            return True
        restricted = [points[index] for index in on_plane]
        projected: list[list[Rational]] | None = None
        for axis in range(dim):
            candidate = [
                [point[k] for k in range(dim) if k != axis] for point in restricted
            ]
            reference = candidate[0]
            columns = [
                Matrix([[row[k] - reference[k]] for k in range(dim - 1)])
                for row in candidate[1:]
            ]
            if columns and Matrix.hstack(*columns).rank() == dim - 1:
                projected = candidate
                break
        if projected is None:
            continue
        if _oracle_is_edge(
            projected, dim - 1, on_plane.index(first), on_plane.index(second)
        ):
            return True
    return False


def _oracle_edge_set(polytope: RationalVPolytope) -> set[tuple[str, str]]:
    points = _raw_points(polytope)
    dim = len(polytope.space.axes)
    ids = [vertex.vertex_id for vertex in polytope.vertices]
    found: set[tuple[str, str]] = set()
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if _oracle_is_edge(points, dim, first, second):
                pair = (ids[first], ids[second])
                found.add(tuple(sorted(pair)))
    return found


def _figure_coords(result) -> dict[str, tuple[Fraction, ...]]:
    return {
        vertex.vertex_id: tuple(c.as_fraction() for c in vertex.coordinates)
        for vertex in result.figure.vertices
    }


class TestEdgeKnownAnswer:
    def test_segment_has_one_edge(self) -> None:
        result = polytope_edge_profile(_segment())
        assert result.affine_dimension == 1
        assert result.edge_count == 1
        assert [(edge.endpoint_a, edge.endpoint_b) for edge in result.edges] == [
            ("left", "right")
        ]

    def test_square_has_four_cycle_edges(self) -> None:
        assert _edge_set(_square()) == {
            ("bottom_left", "bottom_right"),
            ("bottom_left", "top_left"),
            ("bottom_right", "top_right"),
            ("top_left", "top_right"),
        }

    def test_cube_has_twelve_edges_of_degree_three(self) -> None:
        edges = _edge_set(_cube())
        assert len(edges) == 12
        degree: dict[str, int] = {}
        for first, second in edges:
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
        assert sorted(degree.values()) == [3] * 8
        # No face or space diagonals: Hamming distance one exactly.
        for first, second in edges:
            coords = _coords(_cube())
            assert (
                sum(a != b for a, b in zip(coords[first], coords[second], strict=True))
                == 1
            )

    def test_tetrahedron_is_complete_graph(self) -> None:
        result = polytope_edge_profile(_tetrahedron())
        assert result.edge_count == 6

    def test_edges_are_sorted_unique_with_endpoint_transport(self) -> None:
        result = polytope_edge_profile(_cube())
        pairs = [(edge.endpoint_a, edge.endpoint_b) for edge in result.edges]
        assert pairs == sorted(pairs)
        assert len(set(pairs)) == len(pairs)
        assert all(first < second for first, second in pairs)
        assert result.edge_count == len(pairs)
        source_ids = {vertex.vertex_id for vertex in result.polytope.vertices}
        assert {endpoint for pair in pairs for endpoint in pair} <= source_ids

    @pytest.mark.parametrize(
        "polytope",
        [_segment(), _square(), _tetrahedron(), _cube()],
    )
    def test_brute_force_oracle_cross_check(self, polytope: RationalVPolytope) -> None:
        assert _edge_set(polytope) == _oracle_edge_set(polytope)

    def test_facet_row_oracle_agrees_on_prism(self) -> None:
        # A second independent path: raw facet rows from the shared geometry
        # backend (not the kernel's bound PrimitiveFacet profile) recover the
        # same square edge set through incidence intersection plus rank.
        points = _raw_points(_square())
        rows = facets_from_points(points, 2)
        assert len(rows) == 4
        containing = [
            {
                row
                for row, (normal, offset) in enumerate(rows)
                if sum(normal[axis] * points[i][axis] for axis in range(2)) == offset
            }
            for i in range(4)
        ]
        ids = [vertex.vertex_id for vertex in _square().vertices]
        found = set()
        for first in range(4):
            for second in range(first + 1, 4):
                common = containing[first] & containing[second]
                members = [points[k] for k in range(4) if common <= containing[k]]
                reference = members[0]
                columns = [
                    Matrix([[point[k] - reference[k]] for k in range(2)])
                    for point in members[1:]
                ]
                rank = Matrix.hstack(*columns).rank() if columns else 0
                if rank == 1:
                    found.add(tuple(sorted((ids[first], ids[second]))))
        assert found == _edge_set(_square())


class TestVertexFigureKnownAnswer:
    def test_square_figure_is_segment(self) -> None:
        result = polytope_vertex_figure(_square(), "bottom_left")
        assert result.source_affine_dimension == 2
        assert result.figure_affine_dimension == 1
        assert result.center_vertex_id == "bottom_left"
        assert _figure_coords(result) == {
            "sec_bottom_right": (Fraction(1, 2), Fraction(0)),
            "sec_top_left": (Fraction(0), Fraction(1, 2)),
        }
        assert [
            (row.source_vertex_id, row.figure_vertex_id) for row in result.vertex_map
        ] == [
            ("bottom_right", "sec_bottom_right"),
            ("top_left", "sec_top_left"),
        ]

    def test_cube_figure_is_triangle(self) -> None:
        result = polytope_vertex_figure(_cube(), "v000")
        assert result.source_affine_dimension == 3
        assert result.figure_affine_dimension == 2
        assert _figure_coords(result) == {
            "sec_v001": (Fraction(0), Fraction(0), Fraction(1, 2)),
            "sec_v010": (Fraction(0), Fraction(1, 2), Fraction(0)),
            "sec_v100": (Fraction(1, 2), Fraction(0), Fraction(0)),
        }

    def test_simplex_figure_is_simplex_of_dimension_minus_one(self) -> None:
        # Triangle figure at a corner is a segment; tetrahedron figure at a
        # corner is a triangle: the simplex drops exactly one dimension.
        triangle = _polytope(("x", "y"), (("a", (0, 0)), ("b", (1, 0)), ("c", (0, 1))))
        tri_figure = polytope_vertex_figure(triangle, "a")
        assert tri_figure.source_affine_dimension == 2
        assert tri_figure.figure_affine_dimension == 1
        assert len(tri_figure.figure.vertices) == 2
        tet_figure = polytope_vertex_figure(_tetrahedron(), "origin")
        assert tet_figure.source_affine_dimension == 3
        assert tet_figure.figure_affine_dimension == 2
        assert len(tet_figure.figure.vertices) == 3

    def test_segment_figure_is_a_point(self) -> None:
        result = polytope_vertex_figure(_segment(), "left")
        assert result.source_affine_dimension == 1
        assert result.figure_affine_dimension == 0
        assert _figure_coords(result) == {"sec_right": (Fraction(1, 2),)}

    def test_figure_ids_are_canonical_sorted_unique_and_bounded(self) -> None:
        result = polytope_vertex_figure(_cube(), "v111")
        figure_ids = [vertex.vertex_id for vertex in result.figure.vertices]
        assert figure_ids == sorted(figure_ids)
        assert len(set(figure_ids)) == len(figure_ids)
        assert all(1 <= len(vid) <= 64 for vid in figure_ids)
        assert figure_ids == [row.figure_vertex_id for row in result.vertex_map]
        assert all(
            row.figure_vertex_id == f"sec_{row.source_vertex_id}"
            for row in result.vertex_map
        )

    def test_midpoint_and_dimension_identities_replay(self) -> None:
        source = _coords(_square())
        result = polytope_vertex_figure(_square(), "top_right")
        figure = _figure_coords(result)
        for row in result.vertex_map:
            midpoint = figure[row.figure_vertex_id]
            center = source["top_right"]
            neighbor = source[row.source_vertex_id]
            assert (
                tuple(2 * m - c for m, c in zip(midpoint, center, strict=True))
                == neighbor
            )
        assert result.figure_affine_dimension == result.source_affine_dimension - 1


class TestVolumeOracle:
    def test_source_volumes_are_exact(self) -> None:
        assert convex_hull_volume(_square()) == CanonicalRational(num=1, den=1)
        assert convex_hull_volume(_cube()) == CanonicalRational(num=1, den=1)
        assert convex_hull_volume(_tetrahedron()) == CanonicalRational(num=1, den=6)

    def test_figure_composes_with_volume_consumer(self) -> None:
        # CONTRIBUTING consumer check: the accepted figure value feeds the
        # real volume consumer through serialization. A vertex figure is
        # lower-dimensional in its ambient space, so its ambient volume is
        # exactly zero.
        result = polytope_vertex_figure(_cube(), "v000")
        payload = result.figure.model_dump_json()
        revived = type(result.figure).model_validate_json(payload)
        assert revived == result.figure
        raw = tuple(
            tuple(coordinate.as_fraction() for coordinate in vertex.coordinates)
            for vertex in revived.vertices
        )
        assert convex_hull_volume(raw) == CanonicalRational(num=0, den=1)

    def test_edge_profile_source_composes_with_volume_consumer(self) -> None:
        result = polytope_edge_profile(_square())
        payload = result.polytope.model_dump_json()
        revived = RationalVPolytope.model_validate_json(payload)
        assert revived == result.polytope
        assert convex_hull_volume(revived) == CanonicalRational(num=1, den=1)


class TestBoundary:
    def test_over_envelope_vertex_count_refused(self) -> None:
        axes = tuple(f"x{i}" for i in range(7))
        binaries = list(combinations_with_binary_vectors(7, 64))
        rows = tuple((f"v{i:02d}", coords) for i, coords in enumerate(binaries))
        polytope = _polytope(axes, rows)
        distinct = len({coords for _, coords in rows})
        assert distinct * math.comb(distinct, 7) > MAX_FACET_SIGN_TESTS
        with pytest.raises(OperationResourceAdmissionError):
            polytope_edge_profile(polytope)

    def test_vertex_figure_over_envelope_refused(self) -> None:
        axes = tuple(f"x{i}" for i in range(7))
        binaries = list(combinations_with_binary_vectors(7, 64))
        rows = tuple((f"v{i:02d}", coords) for i, coords in enumerate(binaries))
        polytope = _polytope(axes, rows)
        with pytest.raises(OperationResourceAdmissionError):
            polytope_vertex_figure(polytope, "v00")

    def test_dimension_zero_source_cannot_form(self) -> None:
        # A dimension-zero source (one vertex) violates the canonical
        # V-polytope count invariant, so the figure domain rejects it at the
        # value boundary before any kernel runs.
        with pytest.raises(ValidationError):
            RationalVPolytope(
                space=RationalCoordinateSpace(axes=("x",)),
                vertices=(
                    RationalPolytopeVertex(
                        vertex_id="only",
                        coordinates=(CanonicalRational(num=0, den=1),),
                    ),
                ),
            )


class TestAdversarial:
    def test_unknown_vertex_id_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_vertex_figure(_square(), "nope")

    def test_redundant_row_carries_no_edges_and_rejects_figure_center(self) -> None:
        # Duplicate trickery: a redundant midpoint on the bottom edge must
        # not change the square's 4-cycle, and it is not a figure center.
        padded = _polytope(
            ("x", "y"),
            (
                ("bottom_left", (0, 0)),
                ("bottom_right", (2, 0)),
                ("mid", (1, 0)),
                ("top_left", (0, 1)),
                ("top_right", (2, 1)),
            ),
        )
        assert _edge_set(padded) == {
            ("bottom_left", "bottom_right"),
            ("bottom_left", "top_left"),
            ("bottom_right", "top_right"),
            ("top_left", "top_right"),
        }
        with pytest.raises(OperationDomainValidationError):
            polytope_vertex_figure(padded, "mid")

    def test_duplicate_coordinates_rejected_at_value_boundary(self) -> None:
        with pytest.raises(ValidationError):
            _polytope(("x",), (("a", (0,)), ("b", (0,))))

    def test_native_rejects_non_polytope(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_edge_profile("not-a-polytope")  # type: ignore[arg-type]
        with pytest.raises(OperationDomainValidationError):
            polytope_vertex_figure("not-a-polytope", "v")  # type: ignore[arg-type]

    def test_native_rejects_empty_vertex_id(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            polytope_vertex_figure(_square(), "")

    def test_lower_dimensional_hull_rejected(self) -> None:
        flat = _polytope(
            ("x", "y"),
            (("a", (0, 0)), ("b", (1, 0)), ("c", (2, 0))),
        )
        with pytest.raises(OperationDomainValidationError):
            polytope_edge_profile(flat)


class TestDefiningInvariant:
    def test_edge_minimal_faces_are_one_dimensional(self) -> None:
        # Every reported edge's minimal face (common-facet intersection)
        # has affine rank exactly one; every non-edge pair has rank != 1.
        for polytope in (_square(), _cube(), _tetrahedron()):
            points = _raw_points(polytope)
            dim = len(polytope.space.axes)
            rows = facets_from_points(points, dim)
            containing = [
                {
                    row
                    for row, (normal, offset) in enumerate(rows)
                    if sum(normal[axis] * points[i][axis] for axis in range(dim))
                    == offset
                }
                for i in range(len(points))
            ]
            edges = _edge_set(polytope)
            ids = [vertex.vertex_id for vertex in polytope.vertices]
            for first in range(len(points)):
                for second in range(first + 1, len(points)):
                    common = containing[first] & containing[second]
                    members = [
                        points[k] for k in range(len(points)) if common <= containing[k]
                    ]
                    reference = members[0]
                    columns = [
                        Matrix([[point[k] - reference[k]] for k in range(dim)])
                        for point in members[1:]
                    ]
                    rank = Matrix.hstack(*columns).rank() if columns else 0
                    pair = tuple(sorted((ids[first], ids[second])))
                    assert (pair in edges) == (rank == 1)

    def test_figure_vertices_lie_on_incident_edges(self) -> None:
        edges = _edge_set(_cube())
        result = polytope_vertex_figure(_cube(), "v000")
        neighbors = {
            second if first == "v000" else first
            for first, second in edges
            if "v000" in (first, second)
        }
        assert {row.source_vertex_id for row in result.vertex_map} == neighbors
        assert len(result.figure.vertices) == len(neighbors)


class TestNativeVsCatalogParity:
    def test_edge_catalog_entry_matches_native(self) -> None:
        request = EdgeProfileRequest(polytope=_square())
        assert compute_polytope_edge_profile(request) == polytope_edge_profile(
            _square()
        )

    def test_figure_catalog_entry_matches_native(self) -> None:
        request = VertexFigureRequest(polytope=_square(), vertex_id="bottom_left")
        assert compute_polytope_vertex_figure(request) == polytope_vertex_figure(
            _square(), "bottom_left"
        )

    def test_operations_are_published(self) -> None:
        ids = {tool.operation_id for tool in TOOLS}
        assert "polytope.rational.edge_profile.compute" in ids
        assert "polytope.rational.vertex_figure.compute" in ids

    def test_strict_json_examples_validate_and_run(self) -> None:
        runners = {
            "polytope.rational.edge_profile.compute": compute_polytope_edge_profile,
            "polytope.rational.vertex_figure.compute": compute_polytope_vertex_figure,
        }
        by_id = {tool.operation_id for tool in TOOLS}
        assert set(runners) <= by_id
        for tool in TOOLS:
            if tool.operation_id not in runners:
                continue
            for example in tool.examples:
                request = tool.request_type.model_validate_json(
                    json.dumps(example.input)
                )
                assert runners[tool.operation_id](request) is not None


def combinations_with_binary_vectors(dim: int, count: int) -> list[tuple[int, ...]]:
    """``count`` distinct binary vectors of length ``dim`` spanning the space.

    The first ``count - dim`` plain binary values keep the low axes moving
    while one high-bit value per axis forces full affine rank.
    """

    assert count >= 2 * dim
    values = list(range(count - dim)) + [(1 << (dim - 1)) | bit for bit in range(dim)]
    assert len(set(values)) == count
    return [tuple((value >> bit) & 1 for bit in range(dim)) for value in values]
