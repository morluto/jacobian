"""Typed wire contracts for incidence structure operations."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json

MAX_POINTS = 100
MAX_BLOCKS = 100
MAX_T = 10
MAX_SUBSETS = 5_000
MAX_PAIRS = 5_000
MAX_MATRIX_CELLS = 10_000
MAX_GRAPH_EDGES = 5_000
MAX_LABEL_BYTES = 1_024
MAX_RESULT_BYTES = 1_000_000
MAX_TRADE_ORDER = 2
MAX_TRADE_DIFFERENCES = MAX_POINTS + MAX_SUBSETS

_CONTAINMENT_RESULT_PASSES = 2
_MAX_CONTAINMENT_TOTAL_WORK_UNITS = 4_000_000
_MAX_TRADE_TOTAL_WORK_UNITS = 5_000_000
_RESULT_ENVELOPE_BYTES = 4_096
_PROFILE_ENTRY_OVERHEAD_BYTES = 64
_TRADE_DIFFERENCE_OVERHEAD_BYTES = 96


class IncidenceStructure(StrictModel):
    """An ordered point axis and an indexed family of finite blocks.

    Point labels and block IDs are unique. Membership inside one block is
    set-valued, while distinct block IDs may carry equal blocks, so repeated
    blocks remain meaningful in incidence multiplicities.
    """

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


def _subset_count(point_count: int, order: int) -> int:
    return comb(point_count, order) if order <= point_count else 0


def _profile_work_units(incidence: IncidenceStructure, order: int) -> int:
    subset_count = _subset_count(len(incidence.points), order)
    generated_block_subsets = sum(
        _subset_count(len(block), order) for block in incidence.blocks
    )
    canonicalization_units = len(incidence.points) * len(incidence.blocks)
    return canonicalization_units + order * (subset_count + generated_block_subsets)


def _label_wire_bytes(points: tuple[str, ...]) -> int:
    return sum(len(encode_strict_json(point)) + 1 for point in points)


def _subset_label_wire_bytes(points: tuple[str, ...], order: int) -> int:
    point_count = len(points)
    if order > point_count:
        return 0
    appearances_per_point = comb(point_count - 1, order - 1)
    return appearances_per_point * _label_wire_bytes(points)


def _incidence_wire_bytes(incidence: IncidenceStructure) -> int:
    return len(encode_strict_json(incidence.model_dump(mode="json")))


def _require_containment_profile_admitted(
    incidence: IncidenceStructure,
    order: int,
) -> None:
    if not 1 <= order <= MAX_T:
        raise ValueError(f"containment-profile order must be between 1 and {MAX_T}")
    subset_count = _subset_count(len(incidence.points), order)
    if subset_count > MAX_SUBSETS:
        raise ValueError("containment profile exceeds the complete subset-count budget")

    total_work = _CONTAINMENT_RESULT_PASSES * _profile_work_units(incidence, order)
    if total_work > _MAX_CONTAINMENT_TOTAL_WORK_UNITS:
        raise ValueError(
            "containment profile exceeds the operation-plus-replay work budget"
        )

    estimated_result_bytes = (
        _incidence_wire_bytes(incidence)
        + _subset_label_wire_bytes(incidence.points, order)
        + subset_count * _PROFILE_ENTRY_OVERHEAD_BYTES
        + (len(incidence.blocks) + 1) * 32
        + _RESULT_ENVELOPE_BYTES
    )
    if estimated_result_bytes > MAX_RESULT_BYTES:
        raise ValueError(
            "containment profile with its retained source exceeds the output budget"
        )


def _require_incidence_trade_admitted(
    left: IncidenceStructure,
    right: IncidenceStructure,
    max_order: int,
) -> None:
    if not 1 <= max_order <= MAX_TRADE_ORDER:
        raise ValueError(
            f"trade comparison order must be between 1 and {MAX_TRADE_ORDER}"
        )
    if left.points != right.points:
        raise ValueError(
            "trade comparison requires the same ordered point axis on both sides"
        )

    subset_counts = tuple(
        _subset_count(len(left.points), order) for order in range(1, max_order + 1)
    )
    if any(count > MAX_SUBSETS for count in subset_counts):
        raise ValueError("trade comparison exceeds the complete subset-count budget")

    work_per_pass = sum(
        _profile_work_units(left, order) + _profile_work_units(right, order)
        for order in range(1, max_order + 1)
    )
    if _CONTAINMENT_RESULT_PASSES * work_per_pass > _MAX_TRADE_TOTAL_WORK_UNITS:
        raise ValueError(
            "trade comparison exceeds the operation-plus-replay work budget"
        )

    subset_label_bytes = sum(
        _subset_label_wire_bytes(left.points, order)
        for order in range(1, max_order + 1)
    )
    estimated_result_bytes = (
        _incidence_wire_bytes(left)
        + _incidence_wire_bytes(right)
        + subset_label_bytes
        + sum(subset_counts) * _TRADE_DIFFERENCE_OVERHEAD_BYTES
        + max_order * 128
        + _RESULT_ENVELOPE_BYTES
    )
    if estimated_result_bytes > MAX_RESULT_BYTES:
        raise ValueError(
            "trade comparison with its retained sources exceeds the output budget"
        )


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
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the complete multiplicity map for every t-subset of "
                "the ordered point axis. Zero-multiplicity subsets are retained. "
                "The request is rejected before enumeration unless the complete "
                "profile, operation work, result replay, and echoed source fit "
                "their declared budgets."
            )
        }
    )

    incidence: IncidenceStructure = Field(
        description=(
            "Indexed finite block family. Equal blocks with different IDs are "
            "counted separately."
        )
    )
    t: StrictInt = Field(
        ge=1,
        le=MAX_T,
        description="Subset order for the complete containment profile.",
    )

    @model_validator(mode="after")
    def require_complete_profile_bounded(self) -> Self:
        _require_containment_profile_admitted(self.incidence, self.t)
        return self


class ContainmentProfileResult(StrictModel):
    """One complete fixed-order profile bound to its indexed block family."""

    incidence: IncidenceStructure
    t: StrictInt = Field(ge=1, le=MAX_T)
    subset_profile: tuple[tuple[tuple[str, ...], StrictInt], ...] = Field(
        max_length=MAX_SUBSETS
    )
    histogram: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        max_length=MAX_BLOCKS + 1
    )
    total_multiplicity: StrictInt = Field(ge=0)
    min_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)
    max_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)
    is_constant: StrictBool
    constant_lambda: StrictInt | None = Field(default=None, ge=0, le=MAX_BLOCKS)

    @model_validator(mode="after")
    def bind_complete_profile_to_source(self) -> Self:
        from jacobian.math.incidence_structures.operations import (
            _containment_profile_data,
        )

        _require_containment_profile_admitted(self.incidence, self.t)
        expected = _containment_profile_data(self.incidence, self.t)
        actual = (
            self.subset_profile,
            self.histogram,
            self.total_multiplicity,
            self.min_multiplicity,
            self.max_multiplicity,
            self.is_constant,
            self.constant_lambda,
        )
        if actual != expected:
            raise ValueError(
                "containment profile does not match the retained incidence source"
            )
        if len(encode_strict_json(self.model_dump(mode="json"))) > MAX_RESULT_BYTES:
            raise ValueError("containment profile exceeds the exact output budget")
        return self


class IncidenceMultiplicityDifference(StrictModel):
    """One nonzero fixed-subset multiplicity difference between two families."""

    subset: tuple[str, ...] = Field(min_length=1, max_length=MAX_TRADE_ORDER)
    left_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)
    right_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)

    @model_validator(mode="after")
    def require_nonzero_difference(self) -> Self:
        if self.left_multiplicity == self.right_multiplicity:
            raise ValueError("a sparse multiplicity difference must be nonzero")
        return self


class IncidenceMomentComparison(StrictModel):
    """Complete sparse difference data for one positive incidence moment."""

    order: StrictInt = Field(ge=1, le=MAX_TRADE_ORDER)
    left_total: StrictInt = Field(ge=0)
    right_total: StrictInt = Field(ge=0)
    differences: tuple[IncidenceMultiplicityDifference, ...] = Field(
        max_length=MAX_TRADE_DIFFERENCES
    )
    equal: StrictBool

    @model_validator(mode="after")
    def bind_equality_to_sparse_differences(self) -> Self:
        if self.equal != (not self.differences):
            raise ValueError("moment equality must match the sparse difference profile")
        return self


class IncidenceTradeRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compare two indexed block families on exactly the same ordered "
                "point axis through max_order (currently 1 or 2). Every positive-"
                "order multiplicity is compared exactly; omitted result entries "
                "have equal, possibly zero, multiplicity. The zeroth block-count "
                "difference is reported separately."
            )
        }
    )

    left: IncidenceStructure
    right: IncidenceStructure
    max_order: StrictInt = Field(
        ge=1,
        le=MAX_TRADE_ORDER,
        description="Largest positive subset order compared exactly.",
    )

    @model_validator(mode="after")
    def require_trade_comparison_bounded(self) -> Self:
        _require_incidence_trade_admitted(self.left, self.right, self.max_order)
        return self


class IncidenceTradeResult(StrictModel):
    """An exact through-order comparison bound to both indexed block families."""

    left: IncidenceStructure
    right: IncidenceStructure
    max_order: StrictInt = Field(ge=1, le=MAX_TRADE_ORDER)
    zeroth_difference: StrictInt = Field(ge=-MAX_BLOCKS, le=MAX_BLOCKS)
    comparisons: tuple[IncidenceMomentComparison, ...] = Field(
        min_length=1,
        max_length=MAX_TRADE_ORDER,
    )
    positive_moments_equal: StrictBool

    @model_validator(mode="after")
    def bind_comparison_to_sources(self) -> Self:
        from jacobian.math.incidence_structures.operations import _incidence_trade_data

        _require_incidence_trade_admitted(self.left, self.right, self.max_order)
        expected = _incidence_trade_data(self.left, self.right, self.max_order)
        actual = (
            self.zeroth_difference,
            self.comparisons,
            self.positive_moments_equal,
        )
        if actual != expected:
            raise ValueError(
                "incidence trade comparison does not match the retained sources"
            )
        if len(encode_strict_json(self.model_dump(mode="json"))) > MAX_RESULT_BYTES:
            raise ValueError(
                "incidence trade comparison exceeds the exact output budget"
            )
        return self


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
