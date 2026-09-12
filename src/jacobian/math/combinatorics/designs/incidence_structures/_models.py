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
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES as MAX_HYPERGRAPH_EDGES,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_POINTS = 100
MAX_BLOCKS = 100
MAX_T = 10
MAX_SUBSETS = 5_000
MAX_PAIRS = 5_000
MAX_MATRIX_CELLS = 10_000
MAX_GRAPH_EDGES = 5_000
MAX_LABEL_BYTES = 1_024
MAX_TRADE_ORDER = MAX_T
MAX_TRADE_DIFFERENCES = MAX_POINTS + MAX_SUBSETS

# A small exact-cover envelope for the first public design constructor. The
# complete candidate triple family is materialized before the maintained
# generalized exact-cover backend runs, so these bounds cover both the
# candidate representation and its deterministic node-by-item scan.
MAX_STEINER_TRIPLE_ORDER = 15
MAX_STEINER_SEARCH_STATES = 100_000
MAX_STEINER_OUTPUT_BYTES = 64 * 1024
_MAX_STEINER_EXACT_COVER_WORK_UNITS = 256 * 100_000 * 64
_MAX_STEINER_INTERMEDIATE_UNITS = 8_192

_MAX_CONTAINMENT_TOTAL_WORK_UNITS = 4_000_000
_MAX_TRADE_TOTAL_WORK_UNITS = 5_000_000


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
        for kind, labels in (
            ("point label", self.points),
            ("block ID", self.block_ids),
        ):
            for label in labels:
                try:
                    encoded = label.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise _validation_error(
                        "label_not_unicode_scalar_text",
                        f"{kind} must contain only valid Unicode scalar values",
                    ) from exc
                if len(encoded) > MAX_LABEL_BYTES:
                    raise _validation_error(
                        "label_exceeds_byte_bound",
                        f"{kind} must use at most {MAX_LABEL_BYTES} UTF-8 bytes",
                    )
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


class SteinerTripleSystemRequest(StrictModel):
    """Construct an STS(v) through bounded exact pair-cover search."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Construct one canonical Steiner triple system of order v. "
                "The order must be congruent to 1 or 3 modulo 6; a bounded "
                "search may return UNKNOWN without a design when its state "
                "budget is exhausted."
            )
        }
    )

    order: StrictInt = Field(
        ge=3,
        le=MAX_STEINER_TRIPLE_ORDER,
        description=(
            "Number of points; must be congruent to 1 or 3 modulo 6 and at "
            f"most {MAX_STEINER_TRIPLE_ORDER}."
        ),
    )
    search_budget: StrictInt = Field(
        default=MAX_STEINER_SEARCH_STATES,
        ge=1,
        le=MAX_STEINER_SEARCH_STATES,
        description=(
            "Maximum exact-cover search states; exhaustion is reported as "
            "UNKNOWN rather than as a failed construction."
        ),
    )

    @model_validator(mode="after")
    def require_necessary_parameters(self) -> Self:
        # Every pair must occur in one triple, hence b=v(v-1)/6 must be an
        # integer; this is the elementary necessary condition v=1 or 3 mod 6.
        if self.order % 6 not in (1, 3):
            raise _validation_error(
                "steiner_order_necessary_condition",
                "a Steiner triple system requires order congruent to 1 or 3 modulo 6",
            )
        return self


class SteinerTripleSystemResult(StrictModel):
    """One exact construction outcome, including bounded non-completion."""

    status: Literal["COMPUTED", "NOT_FOUND", "UNKNOWN"]
    order: StrictInt = Field(ge=3, le=MAX_STEINER_TRIPLE_ORDER)
    design: IncidenceStructure | None = None
    states_explored: StrictInt = Field(ge=0, le=MAX_STEINER_SEARCH_STATES)

    @model_validator(mode="after")
    def require_status_payload(self) -> Self:
        if self.order % 6 not in (1, 3):
            raise _validation_error(
                "steiner_order_necessary_condition",
                "a Steiner triple system requires order congruent to 1 or 3 modulo 6",
            )
        if self.status == "COMPUTED":
            if self.design is None:
                raise _validation_error(
                    "steiner_computed_without_design",
                    "COMPUTED requires an incidence design",
                )
            if len(self.design.points) != self.order:
                raise _validation_error(
                    "steiner_design_order", "design point count must equal order"
                )
            expected_blocks = self.order * (self.order - 1) // 6
            if len(self.design.blocks) != expected_blocks:
                raise _validation_error(
                    "steiner_design_block_count",
                    "design block count must equal v(v-1)/6",
                )
            expected_points = tuple(f"p{point}" for point in range(self.order))
            expected_block_ids = tuple(f"b{index}" for index in range(expected_blocks))
            if self.design.points != expected_points:
                raise _validation_error(
                    "steiner_design_point_axis",
                    "computed designs must use the canonical point axis",
                )
            if self.design.block_ids != expected_block_ids:
                raise _validation_error(
                    "steiner_design_block_axis",
                    "computed designs must use canonical block IDs",
                )
            if any(len(block) != 3 for block in self.design.blocks):
                raise _validation_error(
                    "steiner_block_size",
                    "every Steiner block must contain exactly 3 points",
                )
            point_index = {point: index for index, point in enumerate(expected_points)}
            block_indices = tuple(
                tuple(point_index[point] for point in block)
                for block in self.design.blocks
            )
            if block_indices != tuple(sorted(block_indices)):
                raise _validation_error(
                    "steiner_block_order",
                    "computed blocks must be in canonical lexicographic order",
                )
        elif self.design is not None:
            raise _validation_error(
                "steiner_noncomputed_design",
                "non-COMPUTED outcomes cannot carry a design",
            )
        return self


def _steiner_output_bytes_bound(order: int) -> int:
    """Return a conservative JSON-size bound for the canonical result."""

    block_count = order * (order - 1) // 6
    # Canonical pN/bN labels are at most three bytes in the admitted range.
    return 4_096 + 32 * order + 64 * block_count + 32 * 3 * block_count


def _require_steiner_triple_system_admitted(order: int, search_budget: int) -> None:
    """Admit all materialized construction work before the search starts."""

    if not 3 <= order <= MAX_STEINER_TRIPLE_ORDER:
        raise IncidenceStructureAdmissionError(
            "steiner_order_out_of_range",
            f"Steiner order must be between 3 and {MAX_STEINER_TRIPLE_ORDER}",
        )
    if order % 6 not in (1, 3):
        raise IncidenceStructureAdmissionError(
            "steiner_order_necessary_condition",
            "a Steiner triple system requires order congruent to 1 or 3 modulo 6",
        )
    if not 1 <= search_budget <= MAX_STEINER_SEARCH_STATES:
        raise IncidenceStructureAdmissionError(
            "steiner_search_budget_out_of_range",
            f"Steiner search budget must be between 1 and {MAX_STEINER_SEARCH_STATES}",
        )
    pair_count = comb(order, 2)
    triple_count = comb(order, 3)
    # Match the generalized exact-cover backend's admitted node-by-item scan
    # bound before materializing its canonical instance.
    work_units = (
        search_budget * pair_count * ((max(triple_count, pair_count) + 63) // 64)
    )
    if work_units > _MAX_STEINER_EXACT_COVER_WORK_UNITS:
        raise IncidenceStructureAdmissionError(
            "steiner_work_budget_exceeded",
            "Steiner construction exceeds the exact-cover work budget",
        )
    intermediate_units = triple_count + 3 * triple_count + pair_count
    if intermediate_units > _MAX_STEINER_INTERMEDIATE_UNITS:
        raise IncidenceStructureAdmissionError(
            "steiner_intermediate_budget_exceeded",
            "Steiner construction exceeds its candidate-family allocation budget",
        )
    if _steiner_output_bytes_bound(order) > MAX_STEINER_OUTPUT_BYTES:
        raise IncidenceStructureAdmissionError(
            "steiner_output_budget_exceeded",
            "Steiner construction exceeds its exact output-size budget",
        )


def _subset_count(point_count: int, order: int) -> int:
    return comb(point_count, order) if order <= point_count else 0


def _containment_axes(
    incidence: IncidenceStructure | FiniteHypergraph,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if isinstance(incidence, FiniteHypergraph):
        return incidence.vertices, tuple(members for _, members in incidence.edges)
    return incidence.points, incidence.blocks


def _profile_work_units(
    incidence: IncidenceStructure | FiniteHypergraph, order: int
) -> int:
    points, blocks = _containment_axes(incidence)
    subset_count = _subset_count(len(points), order)
    generated_block_subsets = sum(_subset_count(len(block), order) for block in blocks)
    canonicalization_units = len(points) * len(blocks) + sum(map(len, blocks))
    return canonicalization_units + max(1, order) * (
        subset_count + generated_block_subsets
    )


def _require_containment_profile_admitted(
    incidence: IncidenceStructure | FiniteHypergraph,
    order: int,
) -> None:
    if not 0 <= order <= MAX_T:
        raise IncidenceStructureAdmissionError(
            "containment_order_out_of_range",
            f"containment-profile order must be between 0 and {MAX_T}",
        )
    points, _ = _containment_axes(incidence)
    subset_count = _subset_count(len(points), order)
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
    # Count mathematical label positions, not encoded transport bytes or
    # pointers. Each output subset repeats its actual source labels.
    _, blocks = _containment_axes(incidence)
    ids = (
        incidence.block_ids
        if isinstance(incidence, IncidenceStructure)
        else tuple(edge_id for edge_id, _ in incidence.edges)
    )
    source_labels = (
        sum(map(len, points))
        + sum(map(len, ids))
        + sum(len(label) for block in blocks for label in block)
    )
    subset_labels = subset_count * order * max(map(len, points), default=0)
    histogram_rows = min(subset_count, len(blocks) + 1)
    integer_slots = subset_count + 2 * histogram_rows + 5
    coordinate_slots = (
        sum(map(len, blocks)) + len(points) + len(blocks) + subset_count * order
    )
    if (
        source_labels + subset_labels > 2**26
        or coordinate_slots + integer_slots > 1_048_576
    ):
        raise IncidenceStructureAdmissionError(
            "containment_output_budget_exceeded",
            "containment profile exceeds its label and coordinate allocation bounds",
        )
    # Multiplicities <= indexed block count, total <= rows * block count.
    # The histogram has at most one row per attained multiplicity and subset.
    coefficient_bits = integer_slots * max(
        1, max(subset_count, len(blocks), subset_count * len(blocks)).bit_length()
    )
    if coefficient_bits > 16_777_216:
        raise IncidenceStructureAdmissionError(
            "containment_output_budget_exceeded",
            "containment profile exceeds its exact integer allocation bound",
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


class IncidenceMatrixRequest(StrictModel):
    incidence: IncidenceStructure


class IncidenceMatrixResult(StrictModel):
    points: tuple[str, ...]
    block_ids: tuple[str, ...]
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if self.matrix.row_count != len(self.points) or self.matrix.column_count != len(
            self.block_ids
        ):
            raise _validation_error(
                "incidence_matrix_shape",
                "matrix dimensions must match point and block axes",
            )
        return self


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

    incidence: IncidenceStructure | FiniteHypergraph = Field(
        description=(
            "Indexed finite block family. Equal blocks with different IDs are "
            "counted separately."
        )
    )
    t: StrictInt = Field(
        ge=0,
        le=MAX_T,
        description="Subset order for the complete containment profile.",
    )


class ContainmentProfileResult(StrictModel):
    """One complete fixed-order profile bound to its indexed block family."""

    incidence: IncidenceStructure | FiniteHypergraph
    t: StrictInt = Field(ge=0, le=MAX_T)
    subset_profile: tuple[tuple[tuple[str, ...], StrictInt], ...] = Field(
        max_length=MAX_SUBSETS
    )
    histogram: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        max_length=MAX_HYPERGRAPH_EDGES + 1
    )
    total_multiplicity: StrictInt = Field(ge=0)
    min_multiplicity: StrictInt = Field(ge=0, le=MAX_HYPERGRAPH_EDGES)
    max_multiplicity: StrictInt = Field(ge=0, le=MAX_HYPERGRAPH_EDGES)
    is_constant: StrictBool
    constant_lambda: StrictInt | None = Field(
        default=None, ge=0, le=MAX_HYPERGRAPH_EDGES
    )

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
        incidence: IncidenceStructure | FiniteHypergraph,
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
    axis: Literal["point", "block"]
    labels: tuple[str, ...]
    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        if self.matrix.row_count != len(self.labels) or self.matrix.column_count != len(
            self.labels
        ):
            raise _validation_error(
                "gram_matrix_shape", "Gram matrix must be square on its labelled axis"
            )
        return self
