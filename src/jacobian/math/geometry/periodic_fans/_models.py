"""Typed wire contracts for exact periodic rational fan validation."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
    canonical_rational_component_digits,
)
from jacobian._models import StrictModel

if TYPE_CHECKING:
    from jacobian.math.geometry.periodic_fans._kernel import RecognizedPeriodicFan

MAX_PERIODIC_LATTICE_RANK = 4
MAX_PERIODIC_CELLS = 32
MAX_PERIODIC_VERTICES = 160
MAX_PERIODIC_COORDINATE_DIGITS = 8
MAX_PERIODIC_INDEX_DIGITS = 12
MAX_PERIODIC_OVERLAP_CANDIDATES = 256
MAX_PERIODIC_QUOTIENT_CELLS = 512
MAX_PERIODIC_POLYGON_VERTICES = 12

_ENVELOPE_DESCRIPTION = (
    f"lattice rank at most {MAX_PERIODIC_LATTICE_RANK}, at most "
    f"{MAX_PERIODIC_CELLS} maximal cells, at most {MAX_PERIODIC_VERTICES} "
    f"vertices, at most {MAX_PERIODIC_OVERLAP_CANDIDATES} overlap candidates, "
    f"at most {MAX_PERIODIC_COORDINATE_DIGITS} decimal digits per coordinate, "
    f"and a period index of at most {MAX_PERIODIC_INDEX_DIGITS} digits"
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"geometry.periodic_fan.{reason}", message)


class PeriodicOverlapCandidate(StrictModel):
    """One declared possible overlap of two maximal cells up to a translation."""

    first_cell: StrictInt = Field(
        ge=0, description="The first maximal cell's index in the presentation."
    )
    second_cell: StrictInt = Field(
        ge=0, description="The second maximal cell's index in the presentation."
    )
    translation: tuple[ExactInteger, ...] = Field(
        max_length=MAX_PERIODIC_LATTICE_RANK,
        description=(
            "Integer translation applied to the second cell before intersecting "
            "it with the first; one coordinate per ambient lattice axis."
        ),
    )


class PeriodicFanPresentation(StrictModel):
    """A finite presentation of a rational fan invariant under a period lattice.

    Maximal cells are full-dimensional simplices, with one additional bounded
    case: strictly convex rank-two polygons with at most
    ``MAX_PERIODIC_POLYGON_VERTICES`` vertices. Their listed vertex order
    is counterclockwise around the boundary and begins at the smallest vertex
    index. Polygon faces are its boundary edges, vertices, and whole polygon.
    Vertices are integer points in the
    closed fundamental parallelotope of the period lattice. Mathematical
    recognition (full-rank and integral period lattice, nondegenerate cells, complete and face-to-face
    overlaps, coverage, and Smith-normal-form unimodularity) belongs to the
    owner's exact kernel, never to this structural model.
    """

    lattice_rank: StrictInt = Field(
        ge=1,
        le=MAX_PERIODIC_LATTICE_RANK,
        description=f"Ambient lattice rank n; at most {MAX_PERIODIC_LATTICE_RANK}.",
    )
    period_basis: tuple[tuple[CanonicalRational, ...], ...] = Field(
        max_length=MAX_PERIODIC_LATTICE_RANK,
        description=(
            "Full-rank period lattice basis as integer row vectors; each "
            "component is a canonical rational so a non-integral claim is "
            "rejected explicitly during recognition."
        ),
    )
    vertices: tuple[tuple[ExactInteger, ...], ...] = Field(
        max_length=MAX_PERIODIC_VERTICES,
        description=(
            f"Integer cell vertices in the closed fundamental parallelotope; at "
            f"most {MAX_PERIODIC_VERTICES} vertices with at most "
            f"{MAX_PERIODIC_COORDINATE_DIGITS} decimal digits per coordinate."
        ),
    )
    cells: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_PERIODIC_CELLS,
        description=(
            "Maximal cells as vertex-index tuples, each a simplex of length "
            "lattice_rank + 1, or a bounded rank-two polygon listed counterclockwise "
            "from its smallest index. Cells are sorted lexicographically and distinct."
        ),
    )
    unimodular_cells: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_PERIODIC_CELLS,
        description="Cell indices claimed unimodular (smooth); checked exactly.",
    )
    overlap_candidates: tuple[PeriodicOverlapCandidate, ...] = Field(
        default=(),
        max_length=MAX_PERIODIC_OVERLAP_CANDIDATES,
        description=(
            "Complete finite set of possible overlaps: every pair of listed "
            "cells and adjacent translates whose closed hulls meet must appear."
        ),
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:  # noqa: C901
        rank = self.lattice_rank
        if len(self.period_basis) != rank or any(
            len(row) != rank for row in self.period_basis
        ):
            raise _validation_error(
                "period_basis_shape",
                "period_basis must be a square lattice_rank x lattice_rank matrix",
            )
        if any(
            canonical_rational_component_digits(value) > MAX_PERIODIC_COORDINATE_DIGITS
            for row in self.period_basis
            for value in row
        ):
            raise _validation_error(
                "coordinates_exceed_envelope",
                "period basis components exceed the "
                f"{MAX_PERIODIC_COORDINATE_DIGITS}-digit envelope",
            )
        limit = 10**MAX_PERIODIC_COORDINATE_DIGITS
        for vertex in self.vertices:
            if len(vertex) != rank:
                raise _validation_error(
                    "vertex_length_matches_lattice_rank",
                    "every vertex must have exactly lattice_rank coordinates",
                )
            if any(abs(value) >= limit for value in vertex):
                raise _validation_error(
                    "coordinates_exceed_envelope",
                    "vertex coordinates exceed the "
                    f"{MAX_PERIODIC_COORDINATE_DIGITS}-digit envelope",
                )
        if len(set(self.vertices)) != len(self.vertices):
            raise _validation_error("duplicate_vertex", "vertices must be distinct")
        previous: tuple[int, ...] | None = None
        max_cell_vertices = MAX_PERIODIC_POLYGON_VERTICES if rank == 2 else rank + 1
        for cell in self.cells:
            if len(cell) < rank + 1 or len(cell) > max_cell_vertices:
                raise _validation_error(
                    "cell_dimension_matches_lattice_rank",
                    "maximal cells must be simplices, except for bounded rank-two polygons",
                )
            if any(index < 0 or index >= len(self.vertices) for index in cell):
                raise _validation_error(
                    "cell_vertex_index_out_of_range",
                    "cell vertex indices must address declared vertices",
                )
            if len(set(cell)) != len(cell):
                raise _validation_error(
                    "cell_vertex_indices_distinct",
                    "cell vertex indices must be distinct",
                )
            if len(cell) == rank + 1 and any(
                first >= second for first, second in pairwise(cell)
            ):
                raise _validation_error(
                    "cell_vertex_indices_strictly_increasing",
                    "simplex vertex indices must be strictly increasing",
                )
            if len(cell) > rank + 1 and rank == 2 and cell[0] != min(cell):
                raise _validation_error(
                    "polygon_vertex_order",
                    "polygon order must begin at its smallest vertex index",
                )
            if previous is not None and cell <= previous:
                raise _validation_error(
                    "cells_sorted_and_distinct",
                    "cells must be sorted lexicographically and distinct",
                )
            previous = cell
        if tuple(sorted(set(self.unimodular_cells))) != self.unimodular_cells:
            raise _validation_error(
                "unimodular_cells_sorted_unique",
                "unimodular_cells must be strictly increasing and unique",
            )
        if any(
            index < 0 or index >= len(self.cells) for index in self.unimodular_cells
        ):
            raise _validation_error(
                "unimodular_cell_index_out_of_range",
                "unimodular cell indices must address declared cells",
            )
        if any(len(self.cells[index]) != rank + 1 for index in self.unimodular_cells):
            raise _validation_error(
                "unimodular_cell_not_simplex",
                "unimodularity claims are defined only for simplex cells",
            )
        candidate_keys = tuple(
            (candidate.first_cell, candidate.second_cell, candidate.translation)
            for candidate in self.overlap_candidates
        )
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise _validation_error(
                "overlap_candidates_sorted",
                "overlap candidates must be sorted by (first_cell, second_cell, translation)",
            )
        if len(set(candidate_keys)) != len(candidate_keys):
            raise _validation_error(
                "duplicate_overlap_candidate",
                "overlap candidates must be declared at most once",
            )
        for candidate in self.overlap_candidates:
            if (
                candidate.first_cell < 0
                or candidate.first_cell >= len(self.cells)
                or candidate.second_cell < 0
                or candidate.second_cell >= len(self.cells)
            ):
                raise _validation_error(
                    "overlap_candidate_index_out_of_range",
                    "overlap candidate cell indices must address declared cells",
                )
            if len(candidate.translation) != rank:
                raise _validation_error(
                    "overlap_translation_length",
                    "overlap translations must have exactly lattice_rank coordinates",
                )
        return self


class PeriodicFanValidationRequest(StrictModel):
    """Validate one proposed periodic rational fan presentation."""

    fan: PeriodicFanPresentation = Field(
        description=f"Proposed periodic fan presentation; envelope: {_ENVELOPE_DESCRIPTION}."
    )


class PeriodicQuotientCell(StrictModel):
    """One orbit of faces of the periodic fan under the period lattice."""

    cell_id: StrictInt
    dimension: StrictInt = Field(ge=0)
    representative_cell: StrictInt = Field(ge=0)
    representative_vertices: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_PERIODIC_LATTICE_RANK + 2
    )
    member_count: StrictInt = Field(ge=1)
    stabilizer_rank: StrictInt = Field(ge=0)
    stabilizer_index: StrictInt | None = Field(
        default=None,
        description=(
            "Index of the stabilizer in the period lattice, or null when the "
            "stabilizer is a proper sublattice of infinite index."
        ),
    )
    stabilizer_basis: tuple[tuple[ExactInteger, ...], ...] = Field(
        default=(),
        description=(
            "Ambient integer basis rows of the translation stabilizer sublattice; "
            "empty exactly when the stabilizer is trivial."
        ),
    )

    @model_validator(mode="after")
    def require_stabilizer_shape(self) -> Self:
        if len(self.stabilizer_basis) != self.stabilizer_rank:
            raise _validation_error(
                "stabilizer_rank_matches_basis",
                "stabilizer_rank must equal the number of stabilizer basis rows",
            )
        if self.stabilizer_rank == 0:
            if self.stabilizer_index is not None or self.stabilizer_basis:
                raise _validation_error(
                    "trivial_stabilizer_shape",
                    "a trivial stabilizer carries no basis and no index",
                )
        elif self.stabilizer_index is not None and self.stabilizer_index < 1:
            raise _validation_error(
                "stabilizer_index_positive",
                "a finite stabilizer index must be strictly positive",
            )
        if len(self.representative_vertices) < self.dimension + 1:
            raise _validation_error(
                "quotient_cell_representative_dimension",
                "a quotient representative must have at least dimension + 1 vertices",
            )
        return self


class PeriodicFaceOrbitRow(StrictModel):
    """One maximal-cell face and the quotient cell it belongs to."""

    cell_id: StrictInt = Field(ge=0)
    face_positions: tuple[StrictInt, ...] = Field(min_length=1)
    quotient_cell_id: StrictInt = Field(ge=0)


class PeriodicQuotientFaceRelation(StrictModel):
    """One orbit-closure incidence ``tau <= sigma`` of quotient cells."""

    tau_cell_id: StrictInt = Field(ge=0)
    sigma_cell_id: StrictInt = Field(ge=0)


class PeriodicFanValidationResult(StrictModel):
    """Source-bound validation verdict and, on VALID, the finite quotient data."""

    fan: PeriodicFanPresentation
    status: Literal["VALID", "INVALID"]
    obstruction_code: str | None = Field(default=None)
    obstruction_message: str | None = Field(default=None)
    period_index: StrictInt | None = Field(default=None, ge=1)
    locally_finite: bool | None = Field(
        default=None,
        description=(
            "True when the full-rank period lattice and bounded cells establish "
            "local finiteness without materializing translates."
        ),
    )
    face_to_face: bool | None = Field(default=None)
    covers_fundamental_domain: bool | None = Field(default=None)
    quotient_cells: tuple[PeriodicQuotientCell, ...] = Field(
        default=(), max_length=MAX_PERIODIC_QUOTIENT_CELLS
    )
    face_orbit_rows: tuple[PeriodicFaceOrbitRow, ...] = Field(default=())
    face_relations: tuple[PeriodicQuotientFaceRelation, ...] = Field(default=())

    @model_validator(mode="after")
    def require_bound_outcome(self) -> Self:
        if self.status == "VALID":
            if (
                self.obstruction_code is not None
                or self.obstruction_message is not None
            ):
                raise _validation_error(
                    "valid_has_no_obstruction",
                    "a VALID result must not carry an obstruction",
                )
            if self.period_index is None or not self.quotient_cells:
                raise _validation_error(
                    "valid_requires_quotient",
                    "a VALID result must carry the period index and quotient cells",
                )
            if tuple(cell.cell_id for cell in self.quotient_cells) != tuple(
                range(len(self.quotient_cells))
            ):
                raise _validation_error(
                    "quotient_cell_ids",
                    "quotient cells must carry consecutive ids",
                )
            for relation in self.face_relations:
                if relation.tau_cell_id > relation.sigma_cell_id:
                    raise _validation_error(
                        "quotient_face_relation_order",
                        "quotient face relations must list the face cell first",
                    )
                if relation.sigma_cell_id >= len(
                    self.quotient_cells
                ) or relation.tau_cell_id >= len(self.quotient_cells):
                    raise _validation_error(
                        "quotient_face_relation_range",
                        "quotient face relations must address declared cells",
                    )
        else:
            if self.obstruction_code is None or self.obstruction_message is None:
                raise _validation_error(
                    "invalid_requires_obstruction",
                    "an INVALID result must carry its first obstruction",
                )
            if (
                self.period_index is not None
                or self.quotient_cells
                or self.face_orbit_rows
                or self.face_relations
            ):
                raise _validation_error(
                    "invalid_carries_no_quotient",
                    "an INVALID result must not carry quotient data",
                )
        return self

    @classmethod
    def _from_recognition(
        cls,
        fan: PeriodicFanPresentation,
        *,
        recognized: RecognizedPeriodicFan,
    ) -> Self:
        obstruction = recognized.obstruction
        if obstruction is not None:
            return cls.model_construct(
                fan=fan,
                status="INVALID",
                obstruction_code=obstruction.code,
                obstruction_message=obstruction.message,
                period_index=None,
                locally_finite=None,
                face_to_face=None,
                covers_fundamental_domain=None,
                quotient_cells=(),
                face_orbit_rows=(),
                face_relations=(),
            )
        return cls.model_construct(
            fan=fan,
            status="VALID",
            obstruction_code=None,
            obstruction_message=None,
            period_index=recognized.period_index,
            locally_finite=True,
            face_to_face=True,
            covers_fundamental_domain=True,
            quotient_cells=tuple(
                PeriodicQuotientCell.model_construct(
                    cell_id=cell.cell_id,
                    dimension=cell.dimension,
                    representative_cell=cell.representative_cell,
                    representative_vertices=cell.representative_vertices,
                    member_count=cell.member_count,
                    stabilizer_rank=cell.stabilizer_rank,
                    stabilizer_index=cell.stabilizer_index,
                    stabilizer_basis=cell.stabilizer_basis,
                )
                for cell in recognized.quotient_cells
            ),
            face_orbit_rows=tuple(
                PeriodicFaceOrbitRow.model_construct(
                    cell_id=row.cell_id,
                    face_positions=row.face_positions,
                    quotient_cell_id=row.quotient_cell_id,
                )
                for row in recognized.orbit_rows
            ),
            face_relations=tuple(
                PeriodicQuotientFaceRelation.model_construct(
                    tau_cell_id=tau, sigma_cell_id=sigma
                )
                for tau, sigma in recognized.face_relations
            ),
        )


__all__ = [
    "MAX_PERIODIC_CELLS",
    "MAX_PERIODIC_COORDINATE_DIGITS",
    "MAX_PERIODIC_INDEX_DIGITS",
    "MAX_PERIODIC_LATTICE_RANK",
    "MAX_PERIODIC_OVERLAP_CANDIDATES",
    "MAX_PERIODIC_QUOTIENT_CELLS",
    "MAX_PERIODIC_VERTICES",
    "PeriodicFaceOrbitRow",
    "PeriodicFanPresentation",
    "PeriodicFanValidationRequest",
    "PeriodicFanValidationResult",
    "PeriodicOverlapCandidate",
    "PeriodicQuotientCell",
    "PeriodicQuotientFaceRelation",
]
