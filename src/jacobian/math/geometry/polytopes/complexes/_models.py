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


__all__ = [
    "MAX_COMPLEX_CELLS",
    "MAX_COMPLEX_COORDINATE_DIGITS",
    "MAX_COMPLEX_COVER_RELATIONS",
    "MAX_COMPLEX_DIMENSION",
    "MAX_COMPLEX_FACE_ENUMERATION_WORK",
    "MAX_COMPLEX_INTERSECTION_WORK",
    "MAX_COMPLEX_PAIRWISE_INTERSECTIONS",
    "MAX_COMPLEX_SOURCE_PRESENTATIONS",
    "MAX_COMPLEX_TOTAL_FACES",
    "ComplexFace",
    "ComplexPoint",
    "FaceCoverRelation",
    "MaximalCellRecord",
    "PairwiseIntersectionRecord",
    "PieceAssignment",
    "PieceCompatibilityRow",
    "PiecewiseEvaluationRequest",
    "PiecewiseEvaluationResult",
    "PiecewisePolynomialRequest",
    "PiecewisePolynomialResult",
    "PolytopalComplexClosureRequest",
    "PolytopalComplexClosureResult",
    "SourceCellTransport",
    "SplineSpaceRequest",
    "SplineSpaceResult",
]
