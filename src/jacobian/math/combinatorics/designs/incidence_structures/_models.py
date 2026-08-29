"""Typed wire contracts for incidence structure operations."""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

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
MAX_TRADE_ORDER = MAX_T
MAX_TRADE_DIFFERENCES = MAX_POINTS + MAX_SUBSETS

_MAX_CONTAINMENT_TOTAL_WORK_UNITS = 4_000_000
_MAX_TRADE_TOTAL_WORK_UNITS = 5_000_000
_RESULT_ENVELOPE_BYTES = 4_096
_PROFILE_ENTRY_OVERHEAD_BYTES = 64
_TRADE_DIFFERENCE_OVERHEAD_BYTES = 96


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"incidence_structure.{code}", message)


class IncidenceStructureAdmissionError(ValueError):
    """Native admission failure for incidence-structure operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class IncidenceStructure(StrictModel):
    """An ordered point axis and an indexed family of finite blocks.

    Point labels and block IDs are unique. Membership inside one block is
    set-valued and is canonicalized to point-axis order on construction,
    so blocks with equal members compare equal regardless of input member
    order. Distinct block IDs may carry equal blocks, so repeated blocks
    remain meaningful in incidence multiplicities.
    """

    points: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    block_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_BLOCKS)
    blocks: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_BLOCKS)

    @model_validator(mode="after")
    def require_valid_incidence(self) -> Self:
        if len(set(self.points)) != len(self.points):
            raise _validation_error(
                "point_labels_not_distinct", "point labels must be distinct"
            )
        if len(set(self.block_ids)) != len(self.block_ids):
            raise _validation_error(
                "block_ids_not_distinct", "block IDs must be distinct"
            )
        if len(self.blocks) != len(self.block_ids):
            raise _validation_error(
                "block_count_mismatch", "blocks and block IDs must have same length"
            )
        point_set = set(self.points)
        canonical_blocks: list[tuple[str, ...]] = []
        for block in self.blocks:
            block_members = set(block)
            if len(block_members) != len(block):
                raise _validation_error(
                    "block_members_not_distinct",
                    "duplicate point labels within a block are not allowed",
                )
            for p in block:
                if p not in point_set:
                    raise _validation_error(
                        "undeclared_block_member",
                        "every block member must be a declared point",
                    )
            canonical_blocks.append(
                tuple(point for point in self.points if point in block_members)
            )
        object.__setattr__(self, "blocks", tuple(canonical_blocks))
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
        raise IncidenceStructureAdmissionError(
            "containment_order_out_of_range",
            f"containment-profile order must be between 1 and {MAX_T}",
        )
    subset_count = _subset_count(len(incidence.points), order)
    if subset_count > MAX_SUBSETS:
        raise IncidenceStructureAdmissionError(
            "containment_subset_budget_exceeded",
            "containment profile exceeds the complete subset-count budget",
        )

    total_work = _profile_work_units(incidence, order)
    if total_work > _MAX_CONTAINMENT_TOTAL_WORK_UNITS:
        raise IncidenceStructureAdmissionError(
            "containment_work_budget_exceeded",
            "containment profile exceeds the execution work budget",
        )

    estimated_result_bytes = (
        _incidence_wire_bytes(incidence)
        + _subset_label_wire_bytes(incidence.points, order)
        + subset_count * _PROFILE_ENTRY_OVERHEAD_BYTES
        + (len(incidence.blocks) + 1) * 32
        + _RESULT_ENVELOPE_BYTES
    )
    if estimated_result_bytes > MAX_RESULT_BYTES:
        raise IncidenceStructureAdmissionError(
            "containment_output_budget_exceeded",
            "containment profile with its retained source exceeds the output budget",
        )


def _require_incidence_trade_admitted(
    left: IncidenceStructure,
    right: IncidenceStructure,
    max_order: int,
) -> None:
    if not 1 <= max_order <= MAX_TRADE_ORDER:
        raise IncidenceStructureAdmissionError(
            "trade_order_out_of_range",
            f"trade comparison order must be between 1 and {MAX_TRADE_ORDER}",
        )
    if left.points != right.points:
        raise IncidenceStructureAdmissionError(
            "trade_point_axis_mismatch",
            "trade comparison requires the same ordered point axis on both sides",
        )

    subset_counts = tuple(
        _subset_count(len(left.points), order) for order in range(1, max_order + 1)
    )
    if any(count > MAX_SUBSETS for count in subset_counts):
        raise IncidenceStructureAdmissionError(
            "trade_subset_budget_exceeded",
            "trade comparison exceeds the complete subset-count budget",
        )

    work_per_pass = sum(
        _profile_work_units(left, order) + _profile_work_units(right, order)
        for order in range(1, max_order + 1)
    )
    if work_per_pass > _MAX_TRADE_TOTAL_WORK_UNITS:
        raise IncidenceStructureAdmissionError(
            "trade_work_budget_exceeded",
            "trade comparison exceeds the execution work budget",
        )

    subset_label_bytes = sum(
        _subset_label_wire_bytes(left.points, order)
        for order in range(1, max_order + 1)
    )
    comparison_axis_bytes = max_order * (_label_wire_bytes(left.points) + 16)
    source_pair_copies = max_order + 1
    estimated_result_bytes = (
        source_pair_copies
        * (_incidence_wire_bytes(left) + _incidence_wire_bytes(right))
        + subset_label_bytes
        + sum(subset_counts) * _TRADE_DIFFERENCE_OVERHEAD_BYTES
        + max_order * 128
        + comparison_axis_bytes
        + _RESULT_ENVELOPE_BYTES
    )
    if estimated_result_bytes > MAX_RESULT_BYTES:
        raise IncidenceStructureAdmissionError(
            "trade_output_budget_exceeded",
            "trade comparison with its retained sources exceeds the output budget",
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
                "profile, execution work, and echoed source fit "
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
    def require_structural_summary_consistency(self) -> Self:
        """Validate only relations carried by this result's own fields."""
        if self.is_constant != (self.min_multiplicity == self.max_multiplicity):
            raise _validation_error(
                "containment_constant_mismatch",
                "constant status must agree with the reported extrema",
            )
        expected_lambda = self.min_multiplicity if self.is_constant else None
        if self.constant_lambda != expected_lambda:
            raise _validation_error(
                "containment_lambda_mismatch",
                "constant lambda must agree with the reported extrema",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        incidence: IncidenceStructure,
        order: int,
        data: tuple[
            tuple[tuple[tuple[str, ...], int], ...],
            tuple[tuple[int, int], ...],
            int,
            int,
            int,
            bool,
            int | None,
        ],
    ) -> Self:
        return cls.model_construct(
            incidence=incidence,
            t=order,
            subset_profile=data[0],
            histogram=data[1],
            total_multiplicity=data[2],
            min_multiplicity=data[3],
            max_multiplicity=data[4],
            is_constant=data[5],
            constant_lambda=data[6],
        )


class IncidenceMultiplicityDifference(StrictModel):
    """One nonzero fixed-subset multiplicity difference between two families.

    Subset labels are distinct; the enclosing ``IncidenceMomentComparison``
    owns the point-axis binding and member ordering.
    """

    subset: tuple[str, ...] = Field(min_length=1, max_length=MAX_TRADE_ORDER)
    left_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)
    right_multiplicity: StrictInt = Field(ge=0, le=MAX_BLOCKS)

    @model_validator(mode="after")
    def require_nonzero_difference_with_distinct_labels(self) -> Self:
        if self.left_multiplicity == self.right_multiplicity:
            raise _validation_error(
                "difference_must_be_nonzero",
                "a sparse multiplicity difference must be nonzero",
            )
        if len(set(self.subset)) != len(self.subset):
            raise _validation_error(
                "difference_labels_not_distinct",
                "difference subsets must have distinct labels",
            )
        return self


class IncidenceMomentComparison(StrictModel):
    """Complete sparse difference data for one positive incidence moment.

    Source-bound value: ``left`` and ``right`` are the indexed block families
    being compared on their shared ordered point axis ``points``. Defining
    invariant: replaying the complete containment profiles of both retained
    families at ``order`` reproduces ``left_total`` and ``right_total`` and
    yields exactly ``differences``, in point-axis combination order, as the
    subsets whose multiplicities differ; every omitted subset therefore
    carries equal, possibly zero, multiplicity on both sides.
    """

    left: IncidenceStructure
    right: IncidenceStructure
    points: tuple[str, ...] = Field(min_length=1, max_length=MAX_POINTS)
    order: StrictInt = Field(ge=1, le=MAX_TRADE_ORDER)
    left_total: StrictInt = Field(ge=0)
    right_total: StrictInt = Field(ge=0)
    differences: tuple[IncidenceMultiplicityDifference, ...] = Field(
        max_length=MAX_TRADE_DIFFERENCES
    )
    equal: StrictBool

    @model_validator(mode="after")
    def bind_moment_to_retained_families(self) -> Self:
        if self.equal != (not self.differences):
            raise _validation_error(
                "moment_equality_mismatch",
                "moment equality must match the sparse difference profile",
            )
        if self.left.points != self.points or self.right.points != self.points:
            raise _validation_error(
                "moment_point_axis_mismatch",
                "moment comparison requires both retained families to share "
                "the declared ordered point axis",
            )
        axis_index = {point: index for index, point in enumerate(self.points)}
        previous_indices: tuple[int, ...] | None = None
        seen_subsets: set[tuple[str, ...]] = set()
        for difference in self.differences:
            subset = difference.subset
            if len(subset) != self.order:
                raise _validation_error(
                    "difference_arity_mismatch",
                    "difference subsets must have exactly order labels",
                )
            if len(set(subset)) != len(subset):
                raise _validation_error(
                    "difference_labels_not_distinct",
                    "difference subsets must have distinct labels",
                )
            if any(label not in axis_index for label in subset):
                raise _validation_error(
                    "difference_label_undeclared",
                    "difference subsets must use declared point-axis labels",
                )
            indices = tuple(axis_index[label] for label in subset)
            if list(indices) != sorted(indices):
                raise _validation_error(
                    "difference_axis_order_mismatch",
                    "difference subsets must follow point-axis order",
                )
            if subset in seen_subsets:
                raise _validation_error(
                    "difference_subsets_not_unique", "difference subsets must be unique"
                )
            seen_subsets.add(subset)
            if previous_indices is not None and indices <= previous_indices:
                raise _validation_error(
                    "difference_combination_order_mismatch",
                    "difference rows must follow point-axis combination order",
                )
            previous_indices = indices
        if any(
            difference.left_multiplicity > self.left_total
            or difference.right_multiplicity > self.right_total
            for difference in self.differences
        ):
            raise _validation_error(
                "moment_total_below_difference",
                "moment totals must bound every reported multiplicity",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        left: IncidenceStructure,
        right: IncidenceStructure,
        order: int,
        left_total: int,
        right_total: int,
        differences: tuple[IncidenceMultiplicityDifference, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            points=left.points,
            order=order,
            left_total=left_total,
            right_total=right_total,
            differences=differences,
            equal=not differences,
        )


class IncidenceTradeRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compare two indexed block families on exactly the same ordered "
                "point axis through max_order. Every positive-order multiplicity "
                "is compared exactly; omitted result entries have equal, "
                "possibly zero, multiplicity. Each requested order must fit the "
                "complete subset-count, execution work, and exact-"
                "output budgets; the schema ceiling is only a conservative "
                "fallback shared with containment profiles. The zeroth "
                "block-count difference is reported separately."
            )
        }
    )

    left: IncidenceStructure
    right: IncidenceStructure
    max_order: StrictInt = Field(
        ge=1,
        le=MAX_TRADE_ORDER,
        description=(
            "Largest positive subset order compared exactly. Higher orders are "
            "admitted whenever the cumulative subset-count, work, and output "
            "budgets fit; the schema ceiling is a conservative fallback."
        ),
    )


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
    def require_comparison_shape(self) -> Self:
        if len(self.comparisons) != self.max_order:
            raise _validation_error(
                "trade_comparison_count_mismatch",
                "trade results require one comparison for every requested order",
            )
        if tuple(comparison.order for comparison in self.comparisons) != tuple(
            range(1, self.max_order + 1)
        ):
            raise _validation_error(
                "trade_comparison_order_mismatch",
                "trade comparisons must be in increasing order from one",
            )
        if any(
            comparison.left != self.left or comparison.right != self.right
            for comparison in self.comparisons
        ):
            raise _validation_error(
                "trade_comparison_source_mismatch",
                "each moment comparison must retain the trade sources",
            )
        if self.positive_moments_equal != all(
            comparison.equal for comparison in self.comparisons
        ):
            raise _validation_error(
                "trade_equality_mismatch",
                "positive-moment equality must agree with every comparison",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        left: IncidenceStructure,
        right: IncidenceStructure,
        max_order: int,
        zeroth_difference: int,
        comparisons: tuple[IncidenceMomentComparison, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            max_order=max_order,
            zeroth_difference=zeroth_difference,
            comparisons=comparisons,
            positive_moments_equal=all(comparison.equal for comparison in comparisons),
        )


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
            raise _validation_error(
                "dual_projection_mismatch",
                "dual structural fields must project incidence",
            )
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
            raise _validation_error(
                "restriction_points_undeclared",
                "points must be a subset of the incidence points",
            )
        if not set(self.block_ids) <= set(self.incidence.block_ids):
            raise _validation_error(
                "restriction_blocks_undeclared",
                "block_ids must be a subset of the incidence block IDs",
            )
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
    kind: Literal["derived", "residual"] = "derived"


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
    axis: Literal["point", "block"] = "point"


class GramResult(StrictModel):
    axis: str
    labels: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
