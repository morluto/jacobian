"""Tests for the bounded exact rational polytope volume operation."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.polytope._models import (
    MAX_COMPUTED_FACETS,
    MAX_FACET_INCIDENCES,
    MAX_FACET_SIGN_TESTS,
    MAX_FACET_TOTAL_SIGN_TESTS,
    MAX_FACETS,
    MAX_VERTICES,
    FacetIncidenceRequest,
    FacetIncidenceResult,
    Halfspace,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    PrimitiveFacet,
    Vertex,
)
from jacobian.math.polytope._operations import (
    compute_facet_incidence,
    compute_polytope_volume,
)


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


def _scaled_halfspace(row: Halfspace, factor: int) -> Halfspace:
    """Return the equivalent half-space scaled by a positive integer."""
    return Halfspace(
        coefficients=tuple(
            CanonicalRational(num=str(int(c.num) * factor), den=c.den)
            for c in row.coefficients
        ),
        offset=CanonicalRational(
            num=str(int(row.offset.num) * factor), den=row.offset.den
        ),
    )


def _six_simplex_rows() -> tuple[Halfspace, ...]:
    """The standard six-simplex ``{x >= 0, sum x <= 1}``: seven rows."""
    rows = []
    for axis in range(6):
        coeffs = [(0, 1)] * 6
        coeffs[axis] = (-1, 1)
        rows.append(_h(*coeffs, offset=(0, 1)))
    rows.append(_h(*([(1, 1)] * 6), offset=(1, 1)))
    return tuple(rows)


def _volume_via_vertices(vertices: tuple[Vertex, ...]) -> PolytopeVolumeResult:
    return compute_polytope_volume(PolytopeVolumeRequest(vertices=vertices))


def _volume_via_halfspaces(
    halfspaces: tuple[Halfspace, ...],
) -> PolytopeVolumeResult:
    return compute_polytope_volume(PolytopeVolumeRequest(halfspaces=halfspaces))


def _facet_profile(vertices: tuple[Vertex, ...]) -> FacetIncidenceResult:
    return compute_facet_incidence(FacetIncidenceRequest(vertices=vertices))


class TestFacetIncidence:
    def test_schema_exposes_the_admission_execution_and_replay_budget(self) -> None:
        schema = FacetIncidenceRequest.model_json_schema()

        description = schema["properties"]["vertices"]["description"]
        assert str(MAX_FACET_SIGN_TESTS) in description
        assert str(MAX_FACET_TOTAL_SIGN_TESTS) in description

    def test_schema_publishes_where_the_result_bounds_attach(self) -> None:
        """The facet and incidence caps are enforced exactly on the
        materialized profile of the bounded enumeration -- which request
        admission itself runs -- and the schema must say so rather than
        promise a row-count upper-bound proof that no admission step
        performs."""
        schema = FacetIncidenceRequest.model_json_schema()

        description = schema["properties"]["vertices"]["description"]
        assert f"{MAX_COMPUTED_FACETS}-facet" in description
        assert f"{MAX_FACET_INCIDENCES}-incidence" in description

    def test_unit_square_returns_canonical_complete_source_incidences(self) -> None:
        vertices = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )

        result = _facet_profile(vertices)

        assert result.vertices == vertices
        assert result.dimension == 2
        assert [
            (
                tuple((c.num, c.den) for c in facet.halfspace.coefficients),
                (facet.halfspace.offset.num, facet.halfspace.offset.den),
                facet.source_vertex_indices,
            )
            for facet in result.facets
        ] == [
            ((("-1", "1"), ("0", "1")), ("0", "1"), (0, 3)),
            ((("0", "1"), ("-1", "1")), ("0", "1"), (0, 1)),
            ((("0", "1"), ("1", "1")), ("1", "1"), (2, 3)),
            ((("1", "1"), ("0", "1")), ("1", "1"), (1, 2)),
        ]

    def test_nonsimplicial_pentagonal_prism_merges_coplanar_subfacets(self) -> None:
        base = ((0, 0), (2, 0), (3, 1), (1, 3), (-1, 1))
        vertices = tuple(_v((x, 1), (y, 1), (z, 1)) for z in (0, 1) for x, y in base)

        result = _facet_profile(vertices)

        assert len(result.facets) == 7
        assert sorted(len(facet.source_vertex_indices) for facet in result.facets) == [
            4,
            4,
            4,
            4,
            4,
            5,
            5,
        ]

    def test_duplicate_source_rows_remain_bound_to_every_incident_facet(self) -> None:
        vertices = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
            _v((0, 1), (0, 1)),
        )

        result = _facet_profile(vertices)

        assert len(result.facets) == 4
        assert any(facet.source_vertex_indices == (0, 3, 4) for facet in result.facets)
        assert any(facet.source_vertex_indices == (0, 1, 4) for facet in result.facets)

    def test_duplicate_rows_do_not_create_lower_dimensional_facets(self) -> None:
        vertices = (
            _v((0, 1), (0, 1)),
            _v((0, 1), (0, 1)),
            _v((1, 1), (2, 1)),
            _v((2, 1), (1, 1)),
        )

        result = _facet_profile(vertices)

        assert len(result.facets) == 3
        assert all(len(facet.source_vertex_indices) >= 2 for facet in result.facets)

    def test_result_rejects_missing_facet_and_wrong_source(self) -> None:
        vertices = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )
        result = _facet_profile(vertices)

        with pytest.raises(ValidationError, match="complete canonical"):
            FacetIncidenceResult(
                vertices=vertices,
                dimension=2,
                facets=result.facets[:-1],
            )
        changed_source = (_v((0, 1), (0, 1)), _v((2, 1), (0, 1)), *vertices[2:])
        with pytest.raises(ValidationError, match="complete canonical"):
            FacetIncidenceResult(
                vertices=changed_source,
                dimension=2,
                facets=result.facets,
            )

    def test_facet_profile_composes_unchanged_into_volume_request(self) -> None:
        """Each computed facet's shared half-space value feeds
        ``PolytopeVolumeRequest`` verbatim -- no coefficient reconstruction."""

        square = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )
        cube = tuple(
            _v((x, 1), (y, 1), (z, 1)) for x in (0, 1) for y in (0, 1) for z in (0, 1)
        )
        for vertices, dimension, volume in (
            (square, 2, ("1", "1")),
            (cube, 3, ("1", "1")),
        ):
            result = _facet_profile(vertices)
            composed = compute_polytope_volume(
                PolytopeVolumeRequest(
                    halfspaces=tuple(facet.halfspace for facet in result.facets)
                )
            )
            assert composed.volume == CanonicalRational(num=volume[0], den=volume[1])
            assert composed.dimension == dimension
            assert composed.representation == "halfspaces"

    def test_seven_dimensional_rows_do_not_feed_the_volume_consumer(self) -> None:
        """The volume consumer caps ambient dimension at 6, so a d = 7
        profile's shared half-space rows are typed values that only compose
        into consumers admitting dimension 7."""

        simplex = (
            _v(*((0, 1) for _ in range(7))),
            *(
                _v(*((1 if index == axis else 0, 1) for axis in range(7)))
                for index in range(7)
            ),
        )
        result = _facet_profile(simplex)
        assert result.dimension == 7
        assert {len(facet.halfspace.coefficients) for facet in result.facets} == {7}

        with pytest.raises(ValidationError, match="exceeds the dimension bound"):
            PolytopeVolumeRequest(
                halfspaces=tuple(facet.halfspace for facet in result.facets)
            )

    def test_forged_rescaled_facet_inequality_is_rejected(self) -> None:
        """A positively rescaled supporting inequality leaves the primitive
        canonical form and must fail typed validation."""

        vertices = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )
        result = _facet_profile(vertices)

        with pytest.raises(ValidationError, match="primitive over the integers"):
            PrimitiveFacet(
                halfspace=_scaled_halfspace(result.facets[0].halfspace, 3),
                source_vertex_indices=result.facets[0].source_vertex_indices,
            )

    def test_non_integer_facet_inequality_is_rejected(self) -> None:
        """A rational row whose cleared form is coprime is still not the
        canonical integral supporting inequality."""

        with pytest.raises(ValidationError, match="entries must be integers"):
            PrimitiveFacet(
                halfspace=Halfspace(
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    offset=CanonicalRational(num="1", den="2"),
                ),
                source_vertex_indices=(0, 3),
            )

    def test_zero_normal_facet_inequality_is_rejected(self) -> None:
        """A tautology row is not a supporting inequality even though its
        primitive form is trivially coprime (offset +/-1)."""

        with pytest.raises(ValidationError, match="nonzero normal"):
            PrimitiveFacet(
                halfspace=Halfspace(
                    coefficients=(
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="0", den="1"),
                    ),
                    offset=CanonicalRational(num="1", den="1"),
                ),
                source_vertex_indices=(0, 3),
            )

    def test_lower_dimensional_input_and_work_overflow_reject_before_enumeration(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="not full-dimensional"):
            FacetIncidenceRequest(
                vertices=(
                    _v((0, 1), (0, 1)),
                    _v((1, 1), (0, 1)),
                )
            )
        vertices = (
            _v(*((0, 1),) * 7),
            *(
                _v(*((1 if index == axis else 0, 1) for axis in range(7)))
                for index in range(7)
            ),
            *(_v(*(((index, 1),) * 7)) for index in range(1, 57)),
        )
        with pytest.raises(ValidationError, match="side-test bound"):
            FacetIncidenceRequest(vertices=vertices)

    def test_interior_source_rows_are_admitted_and_bind_no_facet(self) -> None:
        """Reviewer counterexample: a 7-simplex plus eight distinct strict
        interior points (1/k, ..., 1/k) for k = 9..16 needs only
        16*C(16,7) = 183040 candidate-side tests and its exact profile has
        eight facets. Deduplication cannot remove interior rows, so the
        cyclic-polytope upper bound on all distinct rows (440 > 256) must
        not reject this safely bounded request."""
        simplex = (
            _v(*((0, 1) for _ in range(7))),
            *(
                _v(*((1 if index == axis else 0, 1) for axis in range(7)))
                for index in range(7)
            ),
        )
        unpadded = _facet_profile(simplex)
        vertices = simplex + tuple(_v(*(((1, k),) * 7)) for k in range(9, 17))

        result = _facet_profile(vertices)

        assert len(result.facets) == len(unpadded.facets) == 8
        assert {facet.halfspace for facet in result.facets} == {
            facet.halfspace for facet in unpadded.facets
        }
        assert sorted(
            {index for facet in result.facets for index in facet.source_vertex_indices}
        ) == list(range(8))

    def test_cyclic_profile_beyond_the_facet_cap_is_rejected_at_request_admission(
        self,
    ) -> None:
        """The moment-curve polytope with 15 vertices in d = 7 attains the
        upper-bound-theorem count of 330 facets: its bounded enumeration is
        within the 15*C(15,7) side-test budget, and admission materializes
        that enumeration so the profile is rejected against the published
        facet result limit as a typed request error -- not accepted and
        failed only inside execution."""
        vertices = tuple(_v(*((t**k, 1) for k in range(1, 8))) for t in range(1, 16))

        with pytest.raises(
            ValidationError, match=f"{MAX_COMPUTED_FACETS}-facet result bound"
        ):
            FacetIncidenceRequest(vertices=vertices)

    def test_padded_seven_simplex_admits_distinct_candidates_and_binds_every_row(
        self,
    ) -> None:
        simplex = (
            _v(*((0, 1) for _ in range(7))),
            *(
                _v(*((1 if index == axis else 0, 1) for axis in range(7)))
                for index in range(7)
            ),
        )
        vertices = simplex + (_v(*((0, 1) for _ in range(7))),) * 56
        unpadded = _facet_profile(simplex)

        result = _facet_profile(vertices)

        assert len(result.facets) == len(unpadded.facets) == 8
        assert {facet.halfspace for facet in result.facets} == {
            facet.halfspace for facet in unpadded.facets
        }
        assert result.facets[-1].source_vertex_indices == tuple(range(1, 8))
        incident_positions: set[int] = set()
        for facet in result.facets[:-1]:
            excluded = [c.num for c in facet.halfspace.coefficients].index("-1") + 1
            assert facet.source_vertex_indices == tuple(
                position for position in range(64) if position != excluded
            )
            incident_positions.update(facet.source_vertex_indices)
        assert sorted(incident_positions) == list(range(64))
        assert sum(len(facet.source_vertex_indices) for facet in result.facets) == 448

    def test_padded_distinct_interior_rows_are_charged_per_distinct_row(self) -> None:
        """Reviewer counterexample: a 7-simplex plus 13 distinct strict
        interior points (1/k, ..., 1/k) for k = 9..21, padded to 64 rows
        with duplicate copies. The enumeration receives only the m = 21
        distinct rows and executes 21*C(21,7) = 2441880 candidate-side
        tests, so charging every candidate against the raw 64 source rows
        -- 64*C(21,7) = 7441920 tests -- would reject a request whose real
        work fits the published side-test budget purely because of its
        padding representation."""
        from jacobian.math.polytope._operations import _require_facet_preflight

        simplex = (
            _v(*((0, 1) for _ in range(7))),
            *(
                _v(*((1 if index == axis else 0, 1) for axis in range(7)))
                for index in range(7)
            ),
        )
        interior = tuple(_v(*(((1, k),) * 7)) for k in range(9, 22))
        vertices = simplex + interior + (_v(*((0, 1) for _ in range(7))),) * 43

        assert len(vertices) == MAX_VERTICES == 64
        candidate_count = math.comb(21, 7)
        assert candidate_count * len(vertices) > MAX_FACET_SIGN_TESTS
        assert candidate_count * 21 <= MAX_FACET_SIGN_TESTS

        # Admission charges only the rows the enumeration actually
        # side-tests; the padded request is no longer representation-
        # rejected before its bounded enumeration starts.
        _require_facet_preflight(vertices, 7)

    def test_padded_duplicates_admit_when_distinct_row_work_fits_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end flip of the same charging defect at a test-scale
        budget: padding a hull with m = 10 distinct rows to n = 64 rows
        pushes a per-raw-row charge of 64*C(10,2) = 2880 side tests over
        the patched 450-test budget although the enumeration actually
        executes exactly 10*C(10,2) = 450. The full request-validate ->
        execute -> replay path must admit the padded request and bind
        duplicate positions to their incident facets."""
        import jacobian.math.polytope._operations as operations

        square = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )
        interior = tuple(_v((1, k), (1, k)) for k in range(2, 8))
        vertices = square + interior + (_v((0, 1), (0, 1)),) * 54

        assert len(vertices) == MAX_VERTICES
        executed_side_tests = 10 * math.comb(10, 2)
        monkeypatch.setattr(operations, "MAX_FACET_SIGN_TESTS", executed_side_tests)

        result = _facet_profile(vertices)
        unpadded = _facet_profile(square + interior)

        assert len(result.facets) == len(unpadded.facets) == 4
        assert {facet.halfspace for facet in result.facets} == {
            facet.halfspace for facet in unpadded.facets
        }
        # Facets sort lexicographically: x >= 0, y >= 0, y <= 1, x <= 1.
        # Interior rows bind nothing; the origin and its 54 duplicates sit
        # on both lower facets.
        assert result.facets[0].source_vertex_indices == (0, 3, *range(10, 64))
        assert result.facets[1].source_vertex_indices == (0, 1, *range(10, 64))
        assert result.facets[2].source_vertex_indices == (2, 3)
        assert result.facets[3].source_vertex_indices == (1, 2)

    def test_seven_dimensional_counterexample_has_136_simplicial_facets(self) -> None:
        rows = (
            "0010110",
            "1011101",
            "1000100",
            "1001010",
            "0111000",
            "1100001",
            "0010001",
            "0001100",
            "0100010",
            "1001111",
            "1101110",
            "0110101",
            "1110011",
            "0111011",
        )
        vertices = tuple(_v(*((int(bit), 1) for bit in row)) for row in rows)

        result = _facet_profile(vertices)

        assert result.dimension == 7
        assert len(result.facets) == 136
        assert {len(facet.source_vertex_indices) for facet in result.facets} == {7}


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
                    Vertex(
                        coordinates=(
                            CanonicalRational(num=huge, den="1"),
                            _cr0(),
                        )
                    ),
                    Vertex(
                        coordinates=(
                            _cr0(),
                            CanonicalRational(num=huge, den="1"),
                        )
                    ),
                )
            )

    def test_large_but_representable_triangle_is_returned(self):
        """A triangle whose 20,000-digit area still fits is computed."""
        big = format_canonical_integer(10**10000)
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(), _cr0())),
                    Vertex(
                        coordinates=(
                            CanonicalRational(num=big, den="1"),
                            _cr0(),
                        )
                    ),
                    Vertex(
                        coordinates=(
                            _cr0(),
                            CanonicalRational(num=big, den="1"),
                        )
                    ),
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
                    Vertex(coordinates=(CanonicalRational(num="1", den="1"),)),
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

    def test_small_denominator_interval_is_measured_in_digits(self):
        """Admission measures decimal component lengths, not numeric values:
        the interval [0, 1/40000] has only a five-digit volume denominator
        and must be admitted despite the raw denominator value 40000."""
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(_cr0(),)),
                    Vertex(coordinates=(CanonicalRational(num="1", den="40000"),)),
                )
            )
        )
        assert result.volume == CanonicalRational(num="1", den="40000")
        assert result.dimension == 1

    def test_negative_endpoints_are_measured_by_length(self):
        """Signed numerators contribute their digit length, not their value."""
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(coordinates=(CanonicalRational(num="-3", den="1"),)),
                    Vertex(coordinates=(CanonicalRational(num="5", den="1"),)),
                )
            )
        )
        assert result.volume == CanonicalRational(num="8", den="1")

    def test_endpoint_denominator_product_beyond_the_bound_rejected(self):
        """Endpoints 1/A and 1/(A+1) at ~16,401-digit denominators have a
        reduced volume denominator of ~32,802 digits; the product bound
        rejects them instead of leaking a canonical-conversion failure."""

        def endpoint(den: int) -> Vertex:
            return Vertex(
                coordinates=(
                    CanonicalRational(num="1", den=format_canonical_integer(den)),
                )
            )

        big = 10**16400
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(vertices=(endpoint(big), endpoint(big + 1)))

    def test_representable_large_denominator_interval_computed(self):
        """Coprime ~10,001-digit endpoint denominators multiply to a
        20,001-digit volume denominator that fits and must be returned."""

        def endpoint(den: int) -> Vertex:
            return Vertex(
                coordinates=(
                    CanonicalRational(num="1", den=format_canonical_integer(den)),
                )
            )

        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(endpoint(10**10000 + 3), endpoint(10**10000 + 7))
            )
        )
        assert result.dimension == 1
        assert len(result.volume.den) == 20_001

    def test_degenerate_singleton_at_the_coordinate_bound_is_admitted(self):
        """A single vertex at ``1/10^32767`` satisfies the coordinate
        bound; fewer than two distinct coordinates is a degenerate hull of
        exact volume zero, so admission must deduplicate and return before
        applying the interval growth estimate (review counterexample)."""
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=(
                    Vertex(
                        coordinates=(CanonicalRational(num="1", den="1" + "0" * 32767),)
                    ),
                )
            )
        )
        assert result.volume == CanonicalRational(num="0", den="1")
        assert result.dimension == 1

    def test_duplicated_singleton_interval_is_admitted_with_zero_volume(self):
        """Two identical huge-denominator endpoints still describe one
        distinct coordinate; the kernel returns exact zero."""
        point = Vertex(coordinates=(CanonicalRational(num="1", den="1" + "0" * 32767),))
        result = compute_polytope_volume(PolytopeVolumeRequest(vertices=(point, point)))
        assert result.volume == CanonicalRational(num="0", den="1")

    def test_bounded_halfspace_singleton_is_admitted_with_zero_volume(self):
        """``x <= c`` with ``-x <= -c`` derives the single vertex ``c``
        whose hull volume is exact zero even at the coordinate bound."""
        den = "1" + "0" * 32767
        c = CanonicalRational(num="1", den=den)
        result = _volume_via_halfspaces(
            (
                Halfspace(
                    coefficients=(CanonicalRational(num="1", den="1"),), offset=c
                ),
                Halfspace(
                    coefficients=(CanonicalRational(num="-1", den="1"),),
                    offset=CanonicalRational(num="-1", den=den),
                ),
            )
        )
        assert result.volume == CanonicalRational(num="0", den="1")
        assert result.dimension == 1

    def test_two_distinct_endpoints_beyond_the_bound_still_rejected(self):
        """Deduplication must not over-admit: endpoints at
        ``+/- (10^32768 - 1)`` are two distinct coordinates whose reduced
        difference has a 32,769-digit numerator, so the interval growth
        estimate still rejects the request."""

        def endpoint(num: str) -> Vertex:
            return Vertex(coordinates=(CanonicalRational(num=num, den="1"),))

        huge = format_canonical_integer(10**32768 - 1)
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(vertices=(endpoint(huge), endpoint("-" + huge)))


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


class TestTriangulationWideDenominatorBound:
    def test_eight_prime_polygon_denominator_sum_rejected(self):
        """An eight-vertex convex polygon on distinct ~5000-digit prime
        denominators passes any per-vertex estimate but its shoelace sum
        accumulates a common denominator far beyond the canonical bound;
        the triangulation-aware admission rejects it (review
        counterexample shape)."""

        def prime_like(k: int) -> str:
            # Deterministic large denominators (primality not required: the
            # bound is digit-based, and near-coprime denominators realize
            # the same growth).  Built as a string to stay under CPython's
            # int-to-str conversion limit.
            return "1" + "0" * 4980 + str(100001 + 2 * k)

        vertices = []
        rays = ((2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1))
        for i, (a, b) in enumerate(rays):

            def coord(value: int, den: str) -> CanonicalRational:
                if value == 0:
                    return CanonicalRational(num="0", den="1")
                return CanonicalRational(num=format_canonical_integer(value), den=den)

            vertices.append(
                Vertex(
                    coordinates=(
                        coord(a, prime_like(2 * i)),
                        coord(b, prime_like(2 * i + 1)),
                    )
                )
            )
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(vertices=tuple(vertices))


class TestDuplicateVertexAdmission:
    def test_duplicated_corners_cannot_bypass_the_result_bound(self):
        """Duplicating each corner of the 40,000-digit triangle must not
        let an empty triangulation skip result-size admission; the
        deduplicated guard rejects before execution can fail conversion."""
        big = format_canonical_integer(10**20000)
        corners = (
            Vertex(coordinates=(_cr0(), _cr0())),
            Vertex(
                coordinates=(
                    CanonicalRational(num=big, den="1"),
                    _cr0(),
                )
            ),
            Vertex(
                coordinates=(
                    _cr0(),
                    CanonicalRational(num=big, den="1"),
                )
            ),
        )
        duplicated = tuple(vertex for vertex in corners for _ in range(2))
        with pytest.raises(ValueError, match="result bound"):
            PolytopeVolumeRequest(vertices=duplicated)

    def test_duplicated_representable_square_is_still_computed(self):
        """Exact deduplication keeps ordinary duplicate-laden inputs working."""
        square = (
            _v((0, 1), (0, 1)),
            _v((1, 1), (0, 1)),
            _v((1, 1), (1, 1)),
            _v((0, 1), (1, 1)),
        )
        result = compute_polytope_volume(
            PolytopeVolumeRequest(
                vertices=tuple(vertex for vertex in square for _ in range(2))
            )
        )
        assert result.volume == CanonicalRational(num="1", den="1")

    def test_all_duplicate_points_admit_a_trivial_hull_with_zero_volume(self):
        """The hull-work budget counts unique points: 64 copies of one
        six-dimensional point form a trivial hull whose documented volume
        is exact zero, so raw C(64, 6) counting must not reject it."""
        point = tuple((0, 1) for _ in range(6))
        result = _volume_via_vertices(tuple(_v(*point) for _ in range(64)))
        assert result.volume == CanonicalRational(num="0", den="1")
        assert result.dimension == 6

    def test_few_unique_points_among_many_copies_are_admitted(self):
        """32 copies each of two distinct six-dimensional points have a
        degenerate hull of exact zero volume."""
        pair = (
            _v(*(((0, 1),) + ((0, 1),) * 5)),
            _v(*(((1, 1),) + ((0, 1),) * 5)),
        )
        result = _volume_via_vertices(pair * 32)
        assert result.volume == CanonicalRational(num="0", den="1")
        assert result.dimension == 6

    def test_distinct_points_still_exceed_the_hull_budget(self):
        """64 distinct six-dimensional points remain rejected at C(64, 6)."""
        vertices = tuple(
            _v(*((i**k % 97 + i, 1) for k in range(6))) for i in range(1, 65)
        )
        with pytest.raises(ValueError, match="combinatorial bound"):
            PolytopeVolumeRequest(vertices=vertices)


class TestNativeApiAdmission:
    def test_native_rejects_dimension_and_vertex_excess(self):
        """A 7-dimensional simplex exceeds the 6-dimension native bound."""
        from jacobian.math.polytope import convex_hull_volume

        origin = tuple(Fraction(0) for _ in range(7))
        axes = tuple(tuple(Fraction(int(i == j)) for j in range(7)) for i in range(7))
        with pytest.raises(ValueError, match="dimension"):
            convex_hull_volume((origin, *axes))

    def test_native_rejects_unrepresentable_result_growth(self):
        """The native call on a 40,000-digit triangle must be rejected at
        admission instead of leaking a canonical-validation exception."""
        from jacobian.math.polytope import convex_hull_volume

        huge = Fraction(10) ** 20000
        with pytest.raises(ValueError, match="result bound"):
            convex_hull_volume(
                (
                    (Fraction(0), Fraction(0)),
                    (huge, Fraction(0)),
                    (Fraction(0), huge),
                )
            )

    def test_native_still_returns_representable_volumes(self):
        from jacobian.math.polytope import convex_hull_volume

        big = Fraction(10) ** 10000
        area = convex_hull_volume(
            (
                (Fraction(0), Fraction(0)),
                (big, Fraction(0)),
                (Fraction(0), big),
            )
        )
        assert len(area.num) == 20_000

    def test_native_rejects_hull_work_overflow_before_enumeration(self):
        """64 generic six-dimensional points exceed the hull-work bound at
        C(64, 6) = 74,974,368 subsets; the native wrapper rejects exactly
        like ``PolytopeVolumeRequest`` instead of enumerating unguarded."""
        from jacobian.math.polytope import convex_hull_volume

        points = tuple(tuple(Fraction(i**k) for k in range(6)) for i in range(1, 65))
        with pytest.raises(ValueError, match="combinatorial bound"):
            convex_hull_volume(points)

    def test_native_admits_all_duplicate_points_with_zero_volume(self):
        """The native wrapper applies the hull budget to unique points: 64
        copies of one six-dimensional point return exact zero."""
        from jacobian.math.polytope import convex_hull_volume

        value = convex_hull_volume(tuple((Fraction(0),) * 6 for _ in range(64)))
        assert value == CanonicalRational(num="0", den="1")

    def test_native_admits_degenerate_singleton_at_the_coordinate_bound(self):
        """A single ``1/10^32767`` coordinate is a degenerate one-point
        hull of exact zero volume; admission must not apply the interval
        growth estimate before deduplication (review counterexample)."""
        from jacobian.math.polytope import convex_hull_volume

        value = convex_hull_volume(((Fraction(1, 10**32767),),))
        assert value == CanonicalRational(num="0", den="1")
        duplicated = convex_hull_volume(
            ((Fraction(1, 10**32767),), (Fraction(1, 10**32767),))
        )
        assert duplicated == CanonicalRational(num="0", den="1")


class TestNonsimpleFaceExtremality:
    def test_edge_midpoint_of_4d_prism_is_not_a_vertex(self) -> None:
        """Reviewer counterexample: the midpoint (e1, 1/2) of a vertical
        edge of conv(+/-e1, +/-e2, +/-e3) x [0,1] lies on four maximal
        facets, so incident-facet counting kept it; active-normal rank
        (3 < 4) identifies it as non-extreme, and the exact volume is
        that of the prism, 2^3/3! = 4/3."""
        vertices = []
        for index in range(3):
            for sign in (1, -1):
                for t in ("0", "1"):
                    coords = [(0, 1)] * 4
                    coords[index] = (sign, 1)
                    coords[3] = (int(t), 1)
                    vertices.append(_v(*coords))
        midpoint = _v((1, 1), (0, 1), (0, 1), (1, 2))
        result = _volume_via_vertices((*vertices, midpoint))
        assert result.dimension == 4
        assert result.volume == CanonicalRational(num="4", den="3")


class TestNonzeroNormalContractPublished:
    """The nonzero-normal requirement must be schema-visible and enforced
    by a typed admission error (review thread: publish or remove it)."""

    def test_zero_normal_row_rejected_at_request_admission(self) -> None:
        """The reviewer's tautology `0*x + 0*y <= 1` fails typed validation,
        not a host exception after acceptance."""
        with pytest.raises(ValueError, match="must not all be zero"):
            PolytopeVolumeRequest(
                halfspaces=(
                    _h((1, 1), (0, 1), offset=(1, 1)),
                    _h((-1, 1), (0, 1), offset=(0, 1)),
                    _h((0, 1), (1, 1), offset=(1, 1)),
                    _h((0, 1), (-1, 1), offset=(0, 1)),
                    _h((0, 1), (0, 1), offset=(1, 1)),
                )
            )

    def test_nonzero_normal_rule_is_schema_visible(self) -> None:
        schema = Halfspace.model_json_schema()
        coefficients_description = schema["properties"]["coefficients"]["description"]
        assert "nonzero" in coefficients_description
        request_schema = PolytopeVolumeRequest.model_json_schema()
        halfspaces_description = request_schema["properties"]["halfspaces"][
            "description"
        ]
        assert "nonzero normal" in halfspaces_description

    def test_operation_example_demonstrates_halfspace_input(self) -> None:
        from jacobian.math.polytope._tools import POLYTOPE_OPERATIONS

        operation = next(
            item
            for item in POLYTOPE_OPERATIONS
            if item.operation_id == "polytope.volume.compute"
        )
        names = [e.name for e in operation.examples]
        assert "unit_square_halfspaces" in names


class TestHullWorkBoundPublished:
    """The coupled C(n, d) hull-work bound must be schema-visible so
    clients can size a V-representation per dimension without trial
    execution (review thread: document the coupled hull-work limit)."""

    @staticmethod
    def _expected_max_distinct(dimension: int) -> int:
        from math import comb

        from jacobian.math.polytope._models import MAX_HULL_SUBFACETS

        n = MAX_VERTICES
        while comb(n, dimension) > MAX_HULL_SUBFACETS:
            n -= 1
        return n

    def test_formula_and_threshold_are_schema_visible(self) -> None:
        from jacobian.math.polytope._models import MAX_HULL_SUBFACETS

        schema = PolytopeVolumeRequest.model_json_schema()
        vertices_description = schema["properties"]["vertices"]["description"]
        assert f"C(n, d) <= {MAX_HULL_SUBFACETS}" in vertices_description
        model_description = schema["description"]
        assert "C(n, d)" in model_description

    def test_documented_per_dimension_counts_match_the_bound(self) -> None:
        """Every published usable-count figure is exactly the largest n
        with C(n, d) <= the enforced hull-work ceiling."""
        schema = PolytopeVolumeRequest.model_json_schema()
        description = schema["properties"]["vertices"]["description"]
        for d in range(4, 7):
            expected = self._expected_max_distinct(d)
            assert f"{expected} for d = {d}" in description
        flat = self._expected_max_distinct(3)
        assert flat == MAX_VERTICES
        assert f"up to {flat} distinct vertices for d <= 3" in description

    def test_reviewer_boundary_count_is_rejected_with_typed_error(self) -> None:
        """26 distinct six-dimensional points satisfy every visible field
        bound yet exceed C(26, 6) = 230230; the rejection must say why."""
        points = tuple(_v(*((1000 * j + i, 1) for i in range(6))) for j in range(26))
        with pytest.raises(ValueError, match=r"combinatorial bound \(230230"):
            PolytopeVolumeRequest(vertices=points)

    def test_just_above_the_four_dimensional_maximum_rejected(self) -> None:
        points = tuple(_v(*((1000 * j + i, 1) for i in range(4))) for j in range(49))
        with pytest.raises(ValueError, match=r"combinatorial bound \(211876"):
            PolytopeVolumeRequest(vertices=points)


class TestHalfspaceWorkBoundPublished:
    """The H-representation work ceiling must be schema-visible on the
    distinct-row count so callers can size redundant-copy-laden requests
    without trial execution (review thread: publish the coupled limit)."""

    @staticmethod
    def _expected_max_distinct(dimension: int) -> int:
        from math import comb

        from jacobian.math.polytope._models import MAX_BOUNDEDNESS_COMBINATIONS

        m = MAX_FACETS
        while comb(m, dimension) > MAX_BOUNDEDNESS_COMBINATIONS:
            m -= 1
        return m

    def test_formula_and_threshold_are_schema_visible(self) -> None:
        from jacobian.math.polytope._models import MAX_BOUNDEDNESS_COMBINATIONS

        schema = PolytopeVolumeRequest.model_json_schema()
        description = schema["properties"]["halfspaces"]["description"]
        assert f"C(m, d) <= {MAX_BOUNDEDNESS_COMBINATIONS}" in description
        assert "duplicate rows" in description

    def test_documented_per_dimension_counts_match_the_bound(self) -> None:
        """Every published usable-count figure is exactly the largest m
        with C(m, d) <= the enforced boundedness budget."""
        schema = PolytopeVolumeRequest.model_json_schema()
        description = schema["properties"]["halfspaces"]["description"]
        for d in (5, 6):
            expected = self._expected_max_distinct(d)
            assert f"{expected} for d = {d}" in description
        flat = self._expected_max_distinct(4)
        assert flat == MAX_FACETS
        assert f"{flat} distinct half-spaces for d <= 4" in description

    def test_distinct_rows_still_exceed_the_deduplicated_budget(self) -> None:
        """31 genuinely distinct six-dimensional rows satisfy every visible
        field rule yet exceed C(31, 6) = 736281 > 700000 on distinct rows;
        the typed budget error names the published ceiling."""
        base = _six_simplex_rows()
        rows = list(base)
        for t in range(1, 25):
            coeffs = [(0, 1)] * 6
            coeffs[0] = (-1, 1)
            coeffs[1] = (t, 1)
            rows.append(_h(*coeffs, offset=(t + 1, 1)))
        assert len(rows) == 31
        with pytest.raises(ValueError, match=r"boundedness precheck exceeds"):
            PolytopeVolumeRequest(halfspaces=tuple(rows))


class TestHalfspaceDuplicateRowAdmission:
    """Redundant H-representation rows must neither change the boundedness
    decision nor inflate its combinatorial work estimate (review thread: a
    six-dimensional simplex plus repeated redundant copies was rejected at
    raw C(31, 6) although it derives only seven vertices)."""

    def test_clean_six_simplex_volume_is_one_over_720(self) -> None:
        result = _volume_via_halfspaces(_six_simplex_rows())
        assert result.volume == CanonicalRational(num="1", den="720")
        assert result.dimension == 6
        assert result.representation == "halfspaces"

    def test_simplex_plus_24_redundant_copies_is_admitted(self) -> None:
        """Seven defining inequalities plus 24 redundant copies -- exact
        duplicates and positive rescalings interleaved, 31 raw rows with
        ``C(31, 6) = 736281`` above the old raw-row budget -- describe the
        same simplex and must return its exact volume."""
        base = _six_simplex_rows()
        rows = list(base)
        for j in range(24):
            row = base[j % 7]
            if j % 2 == 0:
                rows.append(row)
            else:
                rows.append(_scaled_halfspace(row, 2))
        assert len(rows) == 31
        result = _volume_via_halfspaces(tuple(rows))
        assert result.volume == CanonicalRational(num="1", den="720")
        assert result.dimension == 6
        assert result.representation == "halfspaces"

    def test_deduplication_merges_positive_rescalings_only(self) -> None:
        """Rows identical up to a positive factor collapse onto their first
        occurrence; a different offset or a sign-flipped normal imposes a
        different inequality and is kept."""
        from jacobian.math.polytope._operations import _deduplicate_halfspaces

        rows = (
            _h((1, 2), (0, 1), offset=(3, 2)),  # x/2 <= 3/2, i.e. x <= 3
            _h((2, 1), (0, 1), offset=(6, 1)),  # 2x <= 6, same inequality
            _h((1, 1), (0, 1), offset=(4, 1)),  # x <= 4, other offset
            _h((-1, 1), (0, 1), offset=(-3, 1)),  # -x <= -3, sign-flipped
        )
        unique = _deduplicate_halfspaces(rows)
        assert [rows.index(row) for row in unique] == [0, 2, 3]

    def test_sign_flip_does_not_merge_and_stays_bounded(self) -> None:
        """The unit square survives duplicated and rescaled rows: merging
        must never merge opposite orientations of one axis."""
        unit_square = (
            _h((1, 1), (0, 1), offset=(1, 1)),
            _h((-1, 1), (0, 1), offset=(0, 1)),
            _h((0, 1), (1, 1), offset=(1, 1)),
            _h((0, 1), (-1, 1), offset=(0, 1)),
        )
        padded = (
            *unit_square,
            _scaled_halfspace(unit_square[0], 3),
            unit_square[1],
            unit_square[2],
            _scaled_halfspace(unit_square[3], 5),
        )
        result = _volume_via_halfspaces(padded)
        assert result.volume == CanonicalRational(num="1", den="1")
        assert result.dimension == 2
