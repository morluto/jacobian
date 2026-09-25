"""Request/result models for the exact polytopal-complex closure owner.

The complex is built from a finite nonempty family of maximal bounded
rational polytopes sharing one labelled ambient coordinate space.  The
value returned is the canonical face-closed family together with the
complete incidence and pairwise-intersection accounting.

Conventions fixed by this owner:

* The ambient dimension is at most :data:`MAX_COMPLEX_DIMENSION`.
* The **empty face is admitted**.  It is represented as one face of
  dimension ``-1`` with no vertices, and it belongs to every maximal cell.
  Consequently the published ``f_vector`` is indexed
  ``(f_{-1}, f_0, ..., f_d)`` with ``f_{-1} = 1``.
* Connectivity is defined through nonempty maximal-cell intersections:
  two maximal cells lie in the same component exactly when they share a
  face (possibly through a chain of shared faces).
"""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.polytopes._models import (
    MAX_COMPUTED_FACETS,
    MAX_VERTICES,
    RationalCoordinateSpace,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.values import MAX_RATIONAL_POLYTOPE_DIMENSION
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.polynomials.values import RationalPolynomial

MAX_COMPLEX_CELLS = 16
"""Maximum number of maximal-cell presentations accepted by one closure."""

MAX_COMPLEX_DIMENSION = 4
"""Maximum ambient dimension of the shared labelled coordinate space."""

MAX_COMPLEX_TOTAL_FACES = 1024
"""Maximum number of canonical faces materialized by one closure result."""

MAX_COMPLEX_PAIRWISE_INTERSECTIONS = MAX_COMPLEX_CELLS * (MAX_COMPLEX_CELLS - 1) // 2
"""Maximum number of distinct maximal-cell pairs intersected by one closure."""

MAX_COMPLEX_COVER_RELATIONS = 4096
"""Maximum number of codimension-one face cover relations in one result."""

MAX_COMPLEX_INTERSECTION_WORK = 2_000_000
"""Maximum exact candidate-vertex systems solved across all maximal-cell pairs."""

MAX_COMPLEX_FACE_ENUMERATION_WORK = 200_000
"""Maximum estimated per-cell face-closure work admitted before enumeration."""

MAX_COMPLEX_COORDINATE_DIGITS = 32
"""Per-component source-coordinate height bound for exact intersection work."""

MAX_COMPLEX_SOURCE_PRESENTATIONS = 64
"""Structural ceiling on the raw cell list before native envelope admission."""

MAX_AFFINE_TRANSFORM_WORK = 2_000_000
"""Maximum matrix-coordinate products admitted for affine complex transport."""

MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS = 512
"""Maximum conservative decimal-height bound for transformed coordinates."""

MAX_AFFINE_TRANSFORM_OUTPUT_BYTES = 10 * 1024 * 1024
"""Maximum estimated serialized source, target, and transport output size."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polytopal_complex.{reason}", message)


class ComplexPoint(StrictModel):
    """One exact rational point in the shared ambient coordinate space."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_RATIONAL_POLYTOPE_DIMENSION
    )


class ComplexFace(StrictModel):
    """One canonical face of a polytopal complex.

    A face of dimension ``-1`` is the empty face and carries no vertices.
    Every face of nonnegative dimension carries its complete vertex family
    in the shared ambient coordinates.
    """

    face_id: str = Field(min_length=1, max_length=64)
    dimension: int = Field(ge=-1, le=MAX_COMPLEX_DIMENSION)
    vertices: tuple[ComplexPoint, ...] = Field(max_length=MAX_VERTICES)
    maximal_cell_ids: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )

    @model_validator(mode="after")
    def require_canonical_face(self) -> Self:
        if self.dimension == -1:
            if self.vertices:
                raise _validation_error(
                    "empty_face_vertices",
                    "the empty face must not carry vertices",
                )
        elif not self.vertices:
            raise _validation_error(
                "face_vertices_missing",
                "a nonempty face must carry at least one vertex",
            )
        coordinates = [vertex.coordinates for vertex in self.vertices]
        if len(set(coordinates)) != len(coordinates):
            raise _validation_error(
                "face_vertices_unique", "face vertices must be distinct"
            )
        if tuple(sorted(self.maximal_cell_ids)) != self.maximal_cell_ids:
            raise _validation_error(
                "face_support_order",
                "maximal-cell support IDs must be sorted",
            )
        if len(set(self.maximal_cell_ids)) != len(self.maximal_cell_ids):
            raise _validation_error(
                "face_support_unique", "maximal-cell support IDs must be unique"
            )
        return self


class MaximalCellRecord(StrictModel):
    """One canonical maximal cell with its presentation provenance and facets."""

    cell_id: str = Field(min_length=1, max_length=64)
    source_indices: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_SOURCE_PRESENTATIONS
    )
    dimension: int = Field(ge=1, le=MAX_COMPLEX_DIMENSION)
    vertices: tuple[ComplexPoint, ...] = Field(min_length=1, max_length=MAX_VERTICES)
    facet_face_ids: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_COMPUTED_FACETS
    )

    @model_validator(mode="after")
    def require_canonical_cell(self) -> Self:
        if tuple(sorted(self.source_indices)) != self.source_indices:
            raise _validation_error(
                "cell_source_order", "source indices must be sorted"
            )
        if len(set(self.source_indices)) != len(self.source_indices):
            raise _validation_error(
                "cell_source_unique", "source indices must be unique"
            )
        if tuple(sorted(set(self.facet_face_ids))) != self.facet_face_ids:
            raise _validation_error(
                "cell_facet_order", "facet face IDs must be sorted and unique"
            )
        coordinates = [vertex.coordinates for vertex in self.vertices]
        if len(set(coordinates)) != len(coordinates):
            raise _validation_error(
                "cell_vertices_unique", "maximal-cell vertices must be distinct"
            )
        return self


class FaceCoverRelation(StrictModel):
    """One codimension-one cover relation between two canonical faces."""

    lower_face_id: str = Field(min_length=1, max_length=64)
    upper_face_id: str = Field(min_length=1, max_length=64)


class PairwiseIntersectionRecord(StrictModel):
    """The exact intersection accounting of one unordered maximal-cell pair."""

    first_cell_id: str = Field(min_length=1, max_length=64)
    second_cell_id: str = Field(min_length=1, max_length=64)
    first_source_index: int = Field(ge=0)
    second_source_index: int = Field(ge=0)
    status: Literal["empty", "face"]
    intersection_face_id: str | None = None

    @model_validator(mode="after")
    def require_status_binding(self) -> Self:
        if self.status == "empty":
            if self.intersection_face_id is not None:
                raise _validation_error(
                    "empty_intersection_face",
                    "an empty intersection must not name a face",
                )
        elif self.intersection_face_id is None:
            raise _validation_error(
                "face_intersection_missing",
                "a face intersection must name the common face",
            )
        return self


class SourceCellTransport(StrictModel):
    """One presentation-position to canonical maximal-cell transport row."""

    source_index: int = Field(ge=0)
    cell_id: str = Field(min_length=1, max_length=64)


class PolytopalComplexClosureResult(StrictModel):
    """A complete canonical face-closed rational polytopal complex.

    ``f_vector`` is indexed ``(f_{-1}, f_0, ..., f_d)`` where ``f_{-1} = 1``
    is the admitted empty face.  ``euler_characteristic`` is the ordinary
    alternating sum over nonempty dimensions, while
    ``reduced_euler_characteristic`` additionally uses the empty-face sign
    and therefore equals ``euler_characteristic - 1``.
    """

    space: RationalCoordinateSpace
    dimension: int = Field(ge=1, le=MAX_COMPLEX_DIMENSION)
    faces: tuple[ComplexFace, ...] = Field(
        min_length=2, max_length=MAX_COMPLEX_TOTAL_FACES
    )
    maximal_cells: tuple[MaximalCellRecord, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )
    cover_relations: tuple[FaceCoverRelation, ...] = Field(
        max_length=MAX_COMPLEX_COVER_RELATIONS
    )
    pairwise_intersections: tuple[PairwiseIntersectionRecord, ...] = Field(
        max_length=MAX_COMPLEX_PAIRWISE_INTERSECTIONS
    )
    source_cell_map: tuple[SourceCellTransport, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_SOURCE_PRESENTATIONS
    )
    f_vector: tuple[int, ...] = Field(min_length=3)
    euler_characteristic: int
    reduced_euler_characteristic: int
    component_count: int = Field(ge=1, le=MAX_COMPLEX_CELLS)
    empty_face_admitted: bool = True

    @model_validator(mode="after")
    def require_canonical_result(self) -> Self:
        if len(self.f_vector) != self.dimension + 2:
            raise _validation_error(
                "f_vector_length",
                "f_vector must carry one entry per dimension from -1 through d",
            )
        if self.f_vector[0] != 1:
            raise _validation_error(
                "f_vector_empty_face",
                "the admitted empty face fixes the first f_vector entry to 1",
            )
        if self.reduced_euler_characteristic != self.euler_characteristic - 1:
            raise _validation_error(
                "reduced_euler_identity",
                "the reduced Euler characteristic must equal the ordinary one minus one",
            )
        face_ids = tuple(face.face_id for face in self.faces)
        if len(set(face_ids)) != len(face_ids):
            raise _validation_error("face_id_unique", "face IDs must be unique")
        cell_ids = tuple(cell.cell_id for cell in self.maximal_cells)
        if cell_ids != tuple(sorted(cell_ids)) or len(set(cell_ids)) != len(cell_ids):
            raise _validation_error(
                "cell_id_order", "maximal-cell IDs must be unique and sorted"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        space: RationalCoordinateSpace,
        dimension: int,
        faces: tuple[ComplexFace, ...],
        maximal_cells: tuple[MaximalCellRecord, ...],
        cover_relations: tuple[FaceCoverRelation, ...],
        pairwise_intersections: tuple[PairwiseIntersectionRecord, ...],
        source_cell_map: tuple[SourceCellTransport, ...],
        f_vector: tuple[int, ...],
        euler_characteristic: int,
        reduced_euler_characteristic: int,
        component_count: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its closure."""

        return cls.model_construct(
            space=space,
            dimension=dimension,
            faces=faces,
            maximal_cells=maximal_cells,
            cover_relations=cover_relations,
            pairwise_intersections=pairwise_intersections,
            source_cell_map=source_cell_map,
            f_vector=f_vector,
            euler_characteristic=euler_characteristic,
            reduced_euler_characteristic=reduced_euler_characteristic,
            component_count=component_count,
            empty_face_admitted=True,
        )


class PieceAssignment(StrictModel):
    cell_id: str = Field(min_length=1, max_length=64)
    polynomial: RationalPolynomial


class PieceCompatibilityRow(StrictModel):
    first_cell_id: str
    second_cell_id: str
    face_id: str
    reduced_difference: RationalPolynomial
    compatible: bool


class PiecewisePolynomialRequest(StrictModel):
    complex: PolytopalComplexClosureResult
    pieces: tuple[PieceAssignment, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )


class PiecewisePolynomialResult(StrictModel):
    complex: PolytopalComplexClosureResult
    pieces: tuple[PieceAssignment, ...]
    compatibility: tuple[PieceCompatibilityRow, ...]
    status: Literal["COMPATIBLE", "INCOMPATIBLE"] = "COMPATIBLE"
    obstruction_face_id: str | None = None
    obstruction_difference: RationalPolynomial | None = None

    @model_validator(mode="after")
    def require_obstruction_shape(self) -> Self:
        if self.status == "INCOMPATIBLE" and (
            self.obstruction_face_id is None or self.obstruction_difference is None
        ):
            raise _validation_error(
                "piece_obstruction",
                "incompatible pieces require a face and reduced difference",
            )
        if self.status == "COMPATIBLE" and (
            self.obstruction_face_id is not None
            or self.obstruction_difference is not None
        ):
            raise _validation_error(
                "piece_obstruction", "compatible pieces must not carry an obstruction"
            )
        return self


class PiecewisePolynomialAdditionRequest(StrictModel):
    """Add two compatible piecewise-polynomial functions on one complex."""

    left: PiecewisePolynomialResult
    right: PiecewisePolynomialResult


class PiecewisePolynomialMultiplicationRequest(StrictModel):
    """Multiply two compatible piecewise-polynomial functions on one complex."""

    left: PiecewisePolynomialResult
    right: PiecewisePolynomialResult


class PiecewisePolynomialScalarMultiplicationRequest(StrictModel):
    """Multiply a compatible piecewise-polynomial function by a rational scalar."""

    function: PiecewisePolynomialResult
    scalar: CanonicalRational


class PiecewiseEvaluationRequest(StrictModel):
    function: PiecewisePolynomialResult
    point: ComplexPoint


class PiecewiseEvaluationResult(StrictModel):
    """An evaluation bound to the complete function that was evaluated.

    Retaining the function (rather than only cell IDs and a scalar) keeps the
    result a closed mathematical value after JSON transport: cell labels are
    local to a complex and do not identify a polynomial or its coordinate
    axes on their own.  The same source is retained for both exact states,
    including ``OUTSIDE_SUPPORT``.
    """

    function: PiecewisePolynomialResult
    status: Literal["EVALUATED", "OUTSIDE_SUPPORT"]
    point: ComplexPoint
    containing_cell_ids: tuple[str, ...]
    value: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_status_shape(self) -> Self:
        if self.status == "EVALUATED":
            if not self.containing_cell_ids or self.value is None:
                raise _validation_error(
                    "evaluation_success_shape",
                    "EVALUATED requires containing cells and an exact value",
                )
        elif self.containing_cell_ids or self.value is not None:
            raise _validation_error(
                "evaluation_outside_shape",
                "OUTSIDE_SUPPORT must not carry containing cells or a value",
            )
        return self


class PiecewiseSmoothnessRequest(StrictModel):
    """Profile exact C^r continuity across the interior facets of a piecewise polynomial."""

    function: PiecewisePolynomialResult
    max_smoothness: int = Field(ge=0, le=4)


class PiecewiseFacetSmoothness(StrictModel):
    first_cell_id: str
    second_cell_id: str
    face_id: str
    smoothness: int = Field(ge=-1, le=4)


class PiecewiseSmoothnessResult(StrictModel):
    function: PiecewisePolynomialResult
    max_smoothness: int = Field(ge=0, le=4)
    facets: tuple[PiecewiseFacetSmoothness, ...]
    smoothness: int = Field(ge=-1, le=4)

    @model_validator(mode="after")
    def require_profile_minimum(self) -> Self:
        if self.smoothness != min(
            (row.smoothness for row in self.facets), default=self.max_smoothness
        ):
            raise _validation_error(
                "smoothness_profile",
                "global smoothness must be the minimum facet smoothness",
            )
        return self


class SplineSpaceRequest(StrictModel):
    complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)


class SplineSpaceResult(StrictModel):
    complex: PolytopalComplexClosureResult
    degree: int
    smoothness: int
    coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...]
    compatibility_matrix: RationalMatrix
    rank: int
    nullity: int
    nullspace_basis: RationalMatrix


class SplineCoordinatesRequest(StrictModel):
    """Express one exact piecewise-polynomial value in a bounded spline basis."""

    function: PiecewisePolynomialResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)


class SplineCoordinatesResult(StrictModel):
    """A source-bound spline space and coordinates of one spline element."""

    spline_space: SplineSpaceResult
    basis_coordinates: tuple[CanonicalRational, ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def require_basis_coordinate_shape(self) -> Self:
        if len(self.basis_coordinates) != self.spline_space.nullity:
            raise _validation_error(
                "spline_basis_coordinates",
                "basis coordinates must match the retained spline nullity",
            )
        return self


class SplineRefinementMapRequest(StrictModel):
    """Compute the exact inclusion induced by a finite complex refinement."""

    coarse: PolytopalComplexClosureResult
    refined: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)


class SplineCellRefinementLineage(StrictModel):
    """One fine cell and its coarse parent through the common overlay."""

    coarse_cell_id: str = Field(min_length=1, max_length=64)
    refined_cell_id: str = Field(min_length=1, max_length=64)
    common_refinement_cell_id: str = Field(min_length=1, max_length=64)


class SplineRefinementMapResult(StrictModel):
    """Exact coefficient-block injection between source-bound spline spaces.

    The columns of ``coarse_nullspace_basis`` are interpreted using
    ``coarse_coefficient_axis``. For each refined top cell, the operation
    copies the polynomial block of its named coarse parent. The resulting
    vectors use ``refined_coefficient_axis`` and lie in the kernel of
    ``refined_compatibility_matrix``.
    """

    coarse_complex: PolytopalComplexClosureResult
    refined_complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)
    coarse_coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...] = Field(
        min_length=1, max_length=4096
    )
    coarse_compatibility_matrix: RationalMatrix
    coarse_rank: int = Field(ge=0)
    coarse_nullspace_basis: RationalMatrix
    refined_coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...] = Field(
        min_length=1, max_length=4096
    )
    refined_compatibility_matrix: RationalMatrix
    refined_rank: int = Field(ge=0)
    refined_nullity: int = Field(ge=0)
    cell_lineage: tuple[SplineCellRefinementLineage, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )

    @model_validator(mode="after")
    def require_refinement_map_binding(self) -> Self:
        source = self.coarse_complex
        target = self.refined_complex
        source_ids = tuple(cell.cell_id for cell in source.maximal_cells)
        target_ids = tuple(cell.cell_id for cell in target.maximal_cells)
        source_width = len(self.coarse_coefficient_axis)
        target_width = len(self.refined_coefficient_axis)
        source_nullity = source_width - self.coarse_rank
        lineage_by_target = {row.refined_cell_id: row for row in self.cell_lineage}
        if (
            source.space != target.space
            or source.dimension != target.dimension
            or tuple(row.refined_cell_id for row in self.cell_lineage) != target_ids
            or len(lineage_by_target) != len(self.cell_lineage)
            or set(lineage_by_target) != set(target_ids)
            or any(
                lineage_by_target[cell_id].common_refinement_cell_id != cell_id
                for cell_id in target_ids
            )
            or self.coarse_compatibility_matrix.column_count != source_width
            or self.coarse_nullspace_basis.column_count != source_width
            or self.coarse_nullspace_basis.row_count != source_nullity
            or self.coarse_rank
            > min(
                self.coarse_compatibility_matrix.row_count,
                self.coarse_compatibility_matrix.column_count,
            )
            or self.refined_compatibility_matrix.column_count != target_width
            or self.refined_rank
            > min(
                self.refined_compatibility_matrix.row_count,
                self.refined_compatibility_matrix.column_count,
            )
            or self.refined_nullity != target_width - self.refined_rank
        ):
            raise _validation_error(
                "spline_refinement_map_binding",
                "spline axes, ranks, and exact cell lineage must bind the retained complexes",
            )
        dimension = len(source.space.axes)
        monomial_count = comb(dimension + self.degree, self.degree)
        if (
            monomial_count * len(source_ids) != source_width
            or monomial_count * len(target_ids) != target_width
        ):
            raise _validation_error(
                "spline_refinement_map_axis",
                "coefficient-axis widths must match the degree and complex cells",
            )
        expected_monomials = _bounded_monomials(dimension, self.degree)
        expected_coarse_axis = tuple(
            (cell_id, monomial)
            for cell_id in source_ids
            for monomial in expected_monomials
        )
        expected_refined_axis = tuple(
            (cell_id, monomial)
            for cell_id in target_ids
            for monomial in expected_monomials
        )
        if (
            self.coarse_coefficient_axis != expected_coarse_axis
            or self.refined_coefficient_axis != expected_refined_axis
            or any(row.coarse_cell_id not in source_ids for row in self.cell_lineage)
        ):
            raise _validation_error(
                "spline_refinement_map_axis",
                "coefficient axes and lineage must be complete and use identical monomials",
            )
        return self


def _bounded_monomials(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Generate total-degree monomials in descending lexicographic order."""
    result: list[tuple[int, ...]] = []

    def extend(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == dimension:
            result.append(prefix)
            return
        for exponent in range(remaining, -1, -1):
            extend((*prefix, exponent), remaining - exponent)

    extend((), degree)
    return tuple(result)


class SplineDimensionRequest(StrictModel):
    """Compute only the exact dimension profile of a bounded spline space."""

    complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)


class SplineDimensionResult(StrictModel):
    """Source-bound compatibility matrix and exact spline dimension data."""

    complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=-1, le=4)
    coefficient_axis: tuple[tuple[str, tuple[int, ...]], ...] = Field(
        min_length=1, max_length=4096
    )
    compatibility_matrix: RationalMatrix
    rank: int = Field(ge=0)
    nullity: int = Field(ge=0)

    @model_validator(mode="after")
    def require_dimension_shape(self) -> Self:
        if self.compatibility_matrix.column_count != len(self.coefficient_axis):
            raise _validation_error(
                "spline_dimension_axis",
                "compatibility matrix columns must match the coefficient axis",
            )
        if self.rank > min(
            self.compatibility_matrix.row_count,
            self.compatibility_matrix.column_count,
        ):
            raise _validation_error(
                "spline_dimension_rank",
                "compatibility matrix rank exceeds its row count",
            )
        if self.nullity != len(self.coefficient_axis) - self.rank:
            raise _validation_error(
                "spline_dimension_nullity",
                "nullity must equal coefficient width minus matrix rank",
            )
        return self


class SplineEvaluationRequest(StrictModel):
    """Evaluate one coefficient vector in the canonical spline basis."""

    complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=0, le=4)
    basis_coefficients: tuple[CanonicalRational, ...] = Field(max_length=4096)
    point: ComplexPoint


class SplineEvaluationResult(StrictModel):
    """Exact evaluation bound to the complex and spline-basis coordinates."""

    complex: PolytopalComplexClosureResult
    degree: int = Field(ge=0, le=12)
    smoothness: int = Field(ge=0, le=4)
    basis_coefficients: tuple[CanonicalRational, ...] = Field(max_length=4096)
    point: ComplexPoint
    containing_cell_ids: tuple[str, ...] = Field(max_length=MAX_COMPLEX_CELLS)
    value: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_evaluation_shape(self) -> Self:
        if self.containing_cell_ids:
            if self.value is None:
                raise _validation_error(
                    "spline_evaluation_value",
                    "a point in the complex requires an exact evaluated value",
                )
        elif self.value is not None:
            raise _validation_error(
                "spline_evaluation_value",
                "a point outside the complex must not carry a value",
            )
        return self


class PolytopalComplexClosureRequest(StrictModel):
    """Compute the canonical face closure of a finite family of maximal cells.

    Every entry must be a labelled bounded rational ``RationalVPolytope``
    that is full-dimensional in the one shared ambient coordinate space;
    mixtures of coordinate spaces are rejected.  The raw list is bounded
    structurally here and admitted against the narrower operation envelope
    by the native kernel.
    """

    cells: tuple[RationalVPolytope, ...] = Field(
        min_length=1,
        max_length=MAX_COMPLEX_SOURCE_PRESENTATIONS,
        description=(
            "Finite nonempty list of maximal bounded rational polytopes in one "
            "shared labelled ambient coordinate space. Duplicate geometries are "
            "merged with presentation provenance retained."
        ),
    )


class CommonRefinementRequest(StrictModel):
    """Refine equal-support, full-dimensional bounded complexes."""

    left: PolytopalComplexClosureResult
    right: PolytopalComplexClosureResult


class CommonRefinementCellPair(StrictModel):
    """Source top cells whose intersection is one top cell of the refinement."""

    left_cell_id: str = Field(min_length=1, max_length=64)
    right_cell_id: str = Field(min_length=1, max_length=64)
    refined_cell_id: str = Field(min_length=1, max_length=64)


class CommonRefinementResult(StrictModel):
    """Canonical overlay complex with explicit source-cell pair provenance."""

    left: PolytopalComplexClosureResult
    right: PolytopalComplexClosureResult
    refinement: PolytopalComplexClosureResult
    cell_pairs: tuple[CommonRefinementCellPair, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )

    @model_validator(mode="after")
    def require_complete_pair_transport(self) -> Self:
        left_ids = {cell.cell_id for cell in self.left.maximal_cells}
        right_ids = {cell.cell_id for cell in self.right.maximal_cells}
        refined_ids = {cell.cell_id for cell in self.refinement.maximal_cells}
        pairs = {(row.left_cell_id, row.right_cell_id) for row in self.cell_pairs}
        if len(pairs) != len(self.cell_pairs):
            raise _validation_error(
                "refinement_pair_unique", "source cell pairs must be unique"
            )
        if any(
            row.left_cell_id not in left_ids
            or row.right_cell_id not in right_ids
            or row.refined_cell_id not in refined_ids
            for row in self.cell_pairs
        ):
            raise _validation_error(
                "refinement_pair_binding",
                "every source pair must bind cells in its complexes",
            )
        if {row.refined_cell_id for row in self.cell_pairs} != refined_ids:
            raise _validation_error(
                "refinement_pair_coverage",
                "every refined maximal cell must have source-pair provenance",
            )
        return self


class ComplexCellTransport(StrictModel):
    """One maximal-cell identity carried through an invertible affine map."""

    source_cell_id: str = Field(min_length=1, max_length=64)
    target_cell_id: str = Field(min_length=1, max_length=64)


class ComplexFaceTransport(StrictModel):
    """One face identity carried through an invertible affine map."""

    source_face_id: str = Field(min_length=1, max_length=64)
    target_face_id: str = Field(min_length=1, max_length=64)


class PolytopalComplexAffineTransformRequest(StrictModel):
    """Apply ``x -> matrix*x + translation`` in the complex's labelled space."""

    complex: PolytopalComplexClosureResult
    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_DIMENSION
    )
    translation: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_DIMENSION
    )

    @model_validator(mode="after")
    def require_square_space_bound_map(self) -> Self:
        dimension = len(self.complex.space.axes)
        if (
            dimension > MAX_COMPLEX_DIMENSION
            or len(self.matrix) != dimension
            or len(self.translation) != dimension
            or any(len(row) != dimension for row in self.matrix)
        ):
            raise _validation_error(
                "affine_transform_shape",
                "matrix and translation dimensions must match the complex axes",
            )
        return self


class PolytopalComplexAffineTransformResult(StrictModel):
    """Affine image with complete source-bound face and maximal-cell maps."""

    source: PolytopalComplexClosureResult
    target: PolytopalComplexClosureResult
    matrix: tuple[tuple[CanonicalRational, ...], ...]
    translation: tuple[CanonicalRational, ...]
    cell_transport: tuple[ComplexCellTransport, ...] = Field(
        min_length=1, max_length=MAX_COMPLEX_CELLS
    )
    face_transport: tuple[ComplexFaceTransport, ...] = Field(
        min_length=2, max_length=MAX_COMPLEX_TOTAL_FACES
    )

    @model_validator(mode="after")
    def require_complete_incidence_transport(self) -> Self:
        dimension = len(self.source.space.axes)
        if (
            len(self.matrix) != dimension
            or len(self.translation) != dimension
            or any(len(row) != dimension for row in self.matrix)
        ):
            raise _validation_error(
                "affine_transform_result_shape",
                "the retained affine map must match the source coordinate axes",
            )
        source_cells = {cell.cell_id: cell for cell in self.source.maximal_cells}
        target_cells = {cell.cell_id: cell for cell in self.target.maximal_cells}
        source_faces = {face.face_id: face for face in self.source.faces}
        target_faces = {face.face_id: face for face in self.target.faces}
        cell_map = {
            row.source_cell_id: row.target_cell_id for row in self.cell_transport
        }
        face_map = {
            row.source_face_id: row.target_face_id for row in self.face_transport
        }
        if (
            self.source.space.axes != self.target.space.axes
            or self.source.dimension != self.target.dimension
            or self.source.f_vector != self.target.f_vector
            or len(cell_map) != len(self.cell_transport)
            or set(cell_map) != set(source_cells)
            or set(cell_map.values()) != set(target_cells)
            or len(face_map) != len(self.face_transport)
            or set(face_map) != set(source_faces)
            or set(face_map.values()) != set(target_faces)
        ):
            raise _validation_error(
                "affine_transform_transport_coverage",
                "affine transport must bijectively cover source and target cells and faces",
            )
        for source_face_id, target_face_id in face_map.items():
            source_face = source_faces[source_face_id]
            target_face = target_faces[target_face_id]
            if source_face.dimension != target_face.dimension or {
                cell_map[cell_id] for cell_id in source_face.maximal_cell_ids
            } != set(target_face.maximal_cell_ids):
                raise _validation_error(
                    "affine_transform_face_incidence",
                    "face transport must preserve dimension and maximal-cell incidence",
                )
        target_covers = {
            (row.lower_face_id, row.upper_face_id)
            for row in self.target.cover_relations
        }
        if {
            (face_map[row.lower_face_id], face_map[row.upper_face_id])
            for row in self.source.cover_relations
        } != target_covers:
            raise _validation_error(
                "affine_transform_cover_incidence",
                "face transport must preserve every cover relation",
            )
        for source_cell_id, target_cell_id in cell_map.items():
            if {
                face_map[face_id]
                for face_id in source_cells[source_cell_id].facet_face_ids
            } != set(target_cells[target_cell_id].facet_face_ids):
                raise _validation_error(
                    "affine_transform_cell_incidence",
                    "cell transport must preserve every facet incidence",
                )

        def pair_profile(
            value: PolytopalComplexClosureResult,
        ) -> dict[tuple[str, str], tuple[str, str | None]]:
            return {
                tuple(sorted((row.first_cell_id, row.second_cell_id))): (
                    row.status,
                    row.intersection_face_id,
                )
                for row in value.pairwise_intersections
            }

        source_pairs = pair_profile(self.source)
        target_pairs = pair_profile(self.target)
        mapped_pairs = {
            tuple(sorted((cell_map[first], cell_map[second]))): (
                status,
                None if face_id is None else face_map[face_id],
            )
            for (first, second), (status, face_id) in source_pairs.items()
        }
        if mapped_pairs != target_pairs:
            raise _validation_error(
                "affine_transform_pairwise_incidence",
                "cell transport must preserve pairwise-intersection incidence",
            )
        return self


__all__ = [
    "MAX_AFFINE_TRANSFORM_COMPONENT_DIGITS",
    "MAX_AFFINE_TRANSFORM_OUTPUT_BYTES",
    "MAX_AFFINE_TRANSFORM_WORK",
    "MAX_COMPLEX_CELLS",
    "MAX_COMPLEX_COORDINATE_DIGITS",
    "MAX_COMPLEX_COVER_RELATIONS",
    "MAX_COMPLEX_DIMENSION",
    "MAX_COMPLEX_FACE_ENUMERATION_WORK",
    "MAX_COMPLEX_INTERSECTION_WORK",
    "MAX_COMPLEX_PAIRWISE_INTERSECTIONS",
    "MAX_COMPLEX_SOURCE_PRESENTATIONS",
    "MAX_COMPLEX_TOTAL_FACES",
    "CommonRefinementCellPair",
    "CommonRefinementRequest",
    "CommonRefinementResult",
    "ComplexCellTransport",
    "ComplexFace",
    "ComplexFaceTransport",
    "ComplexPoint",
    "FaceCoverRelation",
    "MaximalCellRecord",
    "PairwiseIntersectionRecord",
    "PieceAssignment",
    "PieceCompatibilityRow",
    "PiecewiseEvaluationRequest",
    "PiecewiseEvaluationResult",
    "PiecewiseFacetSmoothness",
    "PiecewisePolynomialAdditionRequest",
    "PiecewisePolynomialMultiplicationRequest",
    "PiecewisePolynomialRequest",
    "PiecewisePolynomialResult",
    "PiecewiseSmoothnessRequest",
    "PiecewiseSmoothnessResult",
    "PolytopalComplexAffineTransformRequest",
    "PolytopalComplexAffineTransformResult",
    "PolytopalComplexClosureRequest",
    "PolytopalComplexClosureResult",
    "SourceCellTransport",
    "SplineCellRefinementLineage",
    "SplineCoordinatesRequest",
    "SplineCoordinatesResult",
    "SplineEvaluationRequest",
    "SplineEvaluationResult",
    "SplineRefinementMapRequest",
    "SplineRefinementMapResult",
    "SplineSpaceRequest",
    "SplineSpaceResult",
]
