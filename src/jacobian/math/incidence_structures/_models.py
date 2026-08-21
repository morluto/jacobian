"""Typed wire contracts for incidence structure operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POINTS = 100
MAX_BLOCKS = 100
MAX_T = 10
MAX_SUBSETS = 5_000
MAX_PAIRS = 5_000
MAX_MATRIX_CELLS = 10_000
MAX_GRAPH_EDGES = 5_000
MAX_LABEL_BYTES = 1_024
MAX_RESULT_BYTES = 1_000_000


class IncidenceStructure(StrictModel):
    """A finite incidence structure: points and blocks."""

    points: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    block_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_BLOCKS)
    blocks: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_BLOCKS)

    @model_validator(mode="after")
    def require_valid_incidence(self) -> Self:
        if len(set(self.points)) != len(self.points):
            raise ValueError("point labels must be distinct")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block IDs must be distinct")
        if len(self.blocks) != len(self.block_ids):
            raise ValueError("blocks and block IDs must have same length")
        point_set = set(self.points)
        for block in self.blocks:
            if len(set(block)) != len(block):
                raise ValueError(
                    "duplicate point labels within a block are not allowed"
                )
            for p in block:
                if p not in point_set:
                    raise ValueError("every block member must be a declared point")
        return self


class IncidenceMatrixRequest(StrictModel):
    incidence: IncidenceStructure


class IncidenceMatrixResult(StrictModel):
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]


class DegreeProfileResult(StrictModel):
    """Per-point and per-block degree profiles."""

    point_degrees: tuple[tuple[str, int], ...]
    block_degrees: tuple[tuple[str, int], ...]
    total_incidences: int


# ---------------------------------------------------------------------------
# 3. Containment profiles (t-subset codegree profiles)
# ---------------------------------------------------------------------------


class ContainmentProfileRequest(StrictModel):
    incidence: IncidenceStructure
    t: int = Field(ge=1, le=MAX_T)


class ContainmentProfileResult(StrictModel):
    t: int
    subset_profile: tuple[tuple[tuple[str, ...], int], ...]
    histogram: tuple[tuple[int, int], ...]
    min_multiplicity: int
    max_multiplicity: int
    is_constant: bool
    constant_lambda: int | None = None


# ---------------------------------------------------------------------------
# 4. Block intersection profiles
# ---------------------------------------------------------------------------


class IntersectionsRequest(StrictModel):
    incidence: IncidenceStructure


class IntersectionsResult(StrictModel):
    pairwise: tuple[tuple[str, str, tuple[str, ...], int], ...]
    histogram: tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# 5. Dual incidence structure
# ---------------------------------------------------------------------------


class DualRequest(StrictModel):
    incidence: IncidenceStructure


class DualResult(StrictModel):
    incidence: IncidenceStructure
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]
    point_map: tuple[tuple[str, str], ...]
    block_map: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def require_canonical_projection(self) -> Self:
        if (
            self.points != self.incidence.points
            or self.block_ids != self.incidence.block_ids
            or self.blocks != self.incidence.blocks
        ):
            raise ValueError("dual structural fields must project incidence")
        return self


# ---------------------------------------------------------------------------
# 6. Complement incidence structure
# ---------------------------------------------------------------------------


class ComplementRequest(StrictModel):
    incidence: IncidenceStructure


class ComplementResult(StrictModel):
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]
    correspondence: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]


# ---------------------------------------------------------------------------
# 7. Restriction (point/block deletion and restriction)
# ---------------------------------------------------------------------------


class RestrictionRequest(StrictModel):
    incidence: IncidenceStructure
    points: tuple[str, ...] = Field(default_factory=tuple)
    block_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def require_declared_subsets(self) -> Self:
        if not set(self.points) <= set(self.incidence.points):
            raise ValueError("points must be a subset of the incidence points")
        if not set(self.block_ids) <= set(self.incidence.block_ids):
            raise ValueError("block_ids must be a subset of the incidence block IDs")
        return self


class RestrictionResult(StrictModel):
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]


# ---------------------------------------------------------------------------
# 8. Derived and residual incidence structures
# ---------------------------------------------------------------------------


class DerivedResidualRequest(StrictModel):
    incidence: IncidenceStructure
    point: str
    kind: str = Field(default="derived")

    @model_validator(mode="after")
    def require_valid_kind(self) -> Self:
        if self.kind not in ("derived", "residual"):
            raise ValueError("kind must be 'derived' or 'residual'")
        if self.point not in self.incidence.points:
            raise ValueError(
                "point must be a declared point in the incidence structure"
            )
        return self


class DerivedResidualResult(StrictModel):
    kind: str
    anchor_point: str
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    blocks: tuple[tuple[str, ...], ...]
    source_blocks: tuple[str, ...]


# ---------------------------------------------------------------------------
# 9. Levi graph (bipartite incidence graph)
# ---------------------------------------------------------------------------


class LeviGraphRequest(StrictModel):
    incidence: IncidenceStructure


class LeviGraphResult(StrictModel):
    left_vertices: tuple[str, ...]
    right_vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# 10. Gram / concordance matrix
# ---------------------------------------------------------------------------


class GramRequest(StrictModel):
    incidence: IncidenceStructure
    axis: str = Field(default="point")

    @model_validator(mode="after")
    def require_valid_axis(self) -> Self:
        if self.axis not in ("point", "block"):
            raise ValueError("axis must be 'point' or 'block'")
        return self


class GramResult(StrictModel):
    axis: str
    labels: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
