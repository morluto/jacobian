"""Tests for the bounded exact rational polytope volume operation."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.polytope._models import (
    Halfspace,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    Vertex,
)
from jacobian.math.polytope._operations import compute_polytope_volume


def _cr0() -> CanonicalRational:
    return CanonicalRational(num="0", den="1")


def _v(*coords: tuple[int, int]) -> Vertex:
    """Build a vertex from (num, den) pairs."""
    return Vertex(
        coordinates=tuple({"num": str(num), "den": str(den)} for num, den in coords)
    )


def _h(*coeffs: tuple[int, int], offset: tuple[int, int]) -> Halfspace:
    """Build a half-space ``<a, x> <= b`` from (num, den) pairs."""
    return Halfspace(
        coefficients=tuple({"num": str(num), "den": str(den)} for num, den in coeffs),
        offset={"num": str(offset[0]), "den": str(offset[1])},
    )


def _volume_via_vertices(vertices: tuple[Vertex, ...]) -> PolytopeVolumeResult:
    return compute_polytope_volume(PolytopeVolumeRequest(vertices=vertices))


def _volume_via_halfspaces(
    halfspaces: tuple[Halfspace, ...],
) -> PolytopeVolumeResult:
    return compute_polytope_volume(PolytopeVolumeRequest(halfspaces=halfspaces))


class TestUnitCube:
    def test_unit_square_vertices(self):
        """Unit square [0,1]^2 has volume 1."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1)),
                _v((1, 1), (0, 1)),
                _v((1, 1), (1, 1)),
                _v((0, 1), (1, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 2
        assert result.representation == "vertices"

    def test_unit_cube_vertices(self):
        """Unit cube [0,1]^3 has volume 1."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((1, 1), (0, 1), (0, 1)),
                _v((1, 1), (1, 1), (0, 1)),
                _v((0, 1), (1, 1), (0, 1)),
                _v((0, 1), (0, 1), (1, 1)),
                _v((1, 1), (0, 1), (1, 1)),
                _v((1, 1), (1, 1), (1, 1)),
                _v((0, 1), (1, 1), (1, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 3

    def test_unit_cube_halfspaces(self):
        """Unit cube [0,1]^3 from half-spaces has volume 1."""
        result = _volume_via_halfspaces(
            (
                _h((-1, 1), (0, 1), (0, 1), offset=(0, 1)),
                _h((1, 1), (0, 1), (0, 1), offset=(1, 1)),
                _h((0, 1), (-1, 1), (0, 1), offset=(0, 1)),
                _h((0, 1), (1, 1), (0, 1), offset=(1, 1)),
                _h((0, 1), (0, 1), (-1, 1), offset=(0, 1)),
                _h((0, 1), (0, 1), (1, 1), offset=(1, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 3
        assert result.representation == "halfspaces"


class TestSimplex:
    def test_standard_3simplex(self):
        """Standard 3-simplex has volume 1/6."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((1, 1), (0, 1), (0, 1)),
                _v((0, 1), (1, 1), (0, 1)),
                _v((0, 1), (0, 1), (1, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="6")

    def test_standard_6simplex(self):
        """Standard 6-simplex has volume 1/720."""
        origin = _v(*((0, 1),) * 6)
        basis = tuple(
            _v(*((1, 1) if i == j else (0, 1) for j in range(6))) for i in range(6)
        )
        result = _volume_via_vertices((origin, *basis))
        assert result.volume == CanonicalRational(num="1", den="720")
        assert result.dimension == 6

    def test_simplex_scales_with_side_length(self):
        """A 3-simplex with doubled edges has 8x the volume."""
        small = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((1, 1), (0, 1), (0, 1)),
                _v((0, 1), (1, 1), (0, 1)),
                _v((0, 1), (0, 1), (1, 1)),
            )
        )
        big = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((2, 1), (0, 1), (0, 1)),
                _v((0, 1), (2, 1), (0, 1)),
                _v((0, 1), (0, 1), (2, 1)),
            )
        )
        assert small.volume == CanonicalRational(num="1", den="6")
        assert big.volume == CanonicalRational(num="4", den="3")


class TestRationalVolume:
    def test_pyramid(self):
        """Square pyramid: base area 4, height 1, volume 4/3."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((2, 1), (0, 1), (0, 1)),
                _v((2, 1), (2, 1), (0, 1)),
                _v((0, 1), (2, 1), (0, 1)),
                _v((1, 1), (1, 1), (1, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="4", den="3")

    def test_rational_triangle(self):
        """Triangle with base 2 and height 3 has area 3."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1)),
                _v((2, 1), (0, 1)),
                _v((0, 1), (3, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="3", den="1")

    def test_fractional_pyramid(self):
        """Pyramid with fractional base and height has rational volume."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1), (0, 1)),
                _v((1, 2), (0, 1), (0, 1)),
                _v((1, 2), (1, 2), (0, 1)),
                _v((0, 1), (1, 2), (0, 1)),
                _v((1, 4), (1, 4), (1, 2)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="24")


class TestRejection:
    def test_unbounded_halfspace_representation(self):
        """An unbounded H-representation (no upper bounds) is rejected."""
        with pytest.raises(ValueError):
            _volume_via_halfspaces(
                (
                    _h((-1, 1), (0, 1), offset=(0, 1)),
                    _h((0, 1), (-1, 1), offset=(0, 1)),
                )
            )

    def test_collinear_vertices(self):
        """Collinear vertices are lower-dimensional and have volume zero."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1)),
                _v((1, 1), (1, 1)),
                _v((2, 1), (2, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="0", den="1")
        assert result.dimension == 2

    def test_dimension_exceeds_bound(self):
        """Ambient dimension 7 exceeds the d <= 6 bound."""
        with pytest.raises(ValueError):
            PolytopeVolumeRequest(
                vertices=tuple(
                    _v(*((0, 1),) * 7)
                    if i == 0
                    else _v(*((1, 1) if j == i - 1 else (0, 1) for j in range(7)))
                    for i in range(8)
                )
            )

    def test_work_bound_rejection(self):
        """A large polytope that exceeds the hull work bound is rejected."""
        # 5-cube: 32 vertices, C(32, 5) = 201376 > 200000.
        vertices = tuple(
            _v(*((a, 1), (b, 1), (c, 1), (d, 1), (e, 1)))
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
            for d in (0, 1)
            for e in (0, 1)
        )
        with pytest.raises(ValueError, match="combinatorial bound"):
            _volume_via_vertices(vertices)

    def test_derived_vertex_work_bound_rejected_for_halfspaces(self):
        """An H-representation whose derived vertex set exceeds the hull
        work bound is rejected at request validation.

        The 12 half-spaces of [0,1]^6 enumerate 64 vertices; executing
        would need C(64, 6) = 74,974,368 d-subsets, far beyond the
        combinatorial admission bound.
        """
        halfspaces = []
        for axis in range(6):
            upper = [(0, 1)] * 6
            upper[axis] = (1, 1)
            lower = [(0, 1)] * 6
            lower[axis] = (-1, 1)
            halfspaces.append(_h(*upper, offset=(1, 1)))
            halfspaces.append(_h(*lower, offset=(0, 1)))
        with pytest.raises(ValueError, match="combinatorial bound"):
            PolytopeVolumeRequest(halfspaces=tuple(halfspaces))

    def test_result_carries_only_the_exact_volume(self):
        """The result exposes no generic assurance field."""
        result = _volume_via_vertices(
            (
                _v((0, 1), (0, 1)),
                _v((1, 1), (0, 1)),
                _v((1, 1), (1, 1)),
                _v((0, 1), (1, 1)),
            )
        )
        assert set(result.model_dump()) == {
            "volume",
            "dimension",
            "representation",
        }

    def test_coordinates_whose_volume_leaves_the_canonical_bound_rejected(self):
        """The triangle (0,0),(10^20000,0),(0,10^20000) has a 40,000-digit
        area numerator; admission rejects it instead of failing result
        conversion after acceptance."""

        huge = format_canonical_integer(10**20000)
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(), _cr0())),
                    Vertex(coordinates=(
                        CanonicalRational(num=huge, den="1"),
                        _cr0(),
                    )),
                    Vertex(coordinates=(
                        _cr0(),
                        CanonicalRational(num=huge, den="1"),
                    )),
                )
            )

    def test_large_but_representable_triangle_is_returned(self):
        """A triangle whose 20,000-digit area still fits is computed."""
        big = format_canonical_integer(10**10000)
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(), _cr0())),
                    Vertex(coordinates=(
                        CanonicalRational(num=big, den="1"),
                        _cr0(),
                    )),
                    Vertex(coordinates=(
                        _cr0(),
                        CanonicalRational(num=big, den="1"),
                    )),
                )
            )
        )
        assert len(result.volume.num) == 20_000

    def test_request_schema_advertises_representation_size_bounds(self):
        """The generated schema exposes the vertex/half-space count bounds."""
        import math

        schema = PolytopeVolumeRequest.model_json_schema()
        vertices_schema = schema["properties"]["vertices"]["anyOf"][0]
        halfspaces_schema = schema["properties"]["halfspaces"]["anyOf"][0]
        assert vertices_schema["minItems"] == 1
        assert vertices_schema["maxItems"] == 64
        assert halfspaces_schema["minItems"] == 1
        assert halfspaces_schema["maxItems"] == 64
        assert vertices_schema["maxItems"] >= math.comb(4, 2)


class TestRequestValidation:
    def test_requires_exactly_one_representation(self):
        """Both representations provided is rejected."""
        with pytest.raises(ValueError):
            PolytopeVolumeRequest(
                vertices=(_v((0, 1), (0, 1)),),
                halfspaces=(_h((1, 1), (0, 1), offset=(0, 1)),),
            )

    def test_no_representation_rejected(self):
        """No representation provided is rejected."""
        with pytest.raises(ValueError):
            PolytopeVolumeRequest()

    def test_nonuniform_dimension_rejected(self):
        """Vertices of differing dimension are rejected."""
        with pytest.raises(ValueError):
            PolytopeVolumeRequest(
                vertices=(_v((0, 1), (0, 1)), _v((1, 1), (0, 1), (0, 1))),
            )


class TestDimensionOne:
    def test_interval_vertices(self):
        """The convex hull of 0 and 1 is the advertised d=1 case."""
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(),)),
                    Vertex(coordinates=(
                        CanonicalRational(num="1", den="1"),
                    )),
                )
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 1

    def test_interval_halfspaces(self):
        """x <= 1 and -x <= 0 bound the unit interval."""
        result = _volume_via_halfspaces(
            (
                _h((1, 1), offset=(1, 1)),
                _h((-1, 1), offset=(0, 1)),
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 1


class TestDenominatorGrowth:
    def test_denominator_products_beyond_the_result_bound_rejected(self):
        """Vertices (1/p,1/q),(1/r,1/s) with ~8,500-digit denominators
        produce a reduced area denominator near 34,000 digits; admission
        bounds common-denominator products, not just component length."""

        def small_fraction(den: int) -> Vertex:
            return Vertex(
                coordinates=(
                    CanonicalRational(num="1", den=format_canonical_integer(den)),
                    CanonicalRational(num="1", den=format_canonical_integer(den + 6)),
                )
            )

        big = 10**8500
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(), _cr0())),
                    small_fraction(big),
                    small_fraction(big + 3),
                )
            )


class TestNativeApi:
    def test_domain_exports_the_volume_kernel(self) -> None:
        from jacobian.math import polytope as polytope_module

        assert "convex_hull_volume" in polytope_module.__all__
        assert callable(polytope_module.convex_hull_volume)

    def test_kernel_accepts_mathematical_values(self) -> None:
        from fractions import Fraction

        from jacobian.math.polytope import convex_hull_volume

        area = convex_hull_volume(
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
        )
        assert area == CanonicalRational(num="1", den="2")
        degenerate = convex_hull_volume(
            (
                (Fraction(0),),
                (Fraction(2),),
            )
        )
        assert degenerate == CanonicalRational(num="2", den="1")
