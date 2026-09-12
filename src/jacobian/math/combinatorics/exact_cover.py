"""Canonical values and results for bounded generalized exact cover."""

from __future__ import annotations

import unicodedata
from hashlib import sha256
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.math._labels import OpaqueLabel


def _combinatorics_validation_error(message: str) -> PydanticCustomError:
    lowered = message.lower()
    code = "combinatorics.exact_cover_invariant"
    if "bound" in lowered or "count" in lowered or "limit" in lowered:
        code = "combinatorics.exact_cover_bound"
    return PydanticCustomError(code, message, {})


MAX_EXACT_COVER_ITEMS = 4_096
MAX_EXACT_COVER_PRIMARY_ITEMS = MAX_EXACT_COVER_ITEMS
MAX_EXACT_COVER_SECONDARY_ITEMS = MAX_EXACT_COVER_ITEMS
MAX_EXACT_COVER_ROWS = 4_096
MAX_EXACT_COVER_INCIDENCES = 65_536

# A million-node adversarial pass is too slow to repeat at the public boundary;
# 100,000 nodes per pass is a measured conservative execution fallback,
# independent of the broader 4096-item representation bound.
MAX_EXACT_COVER_SEARCH_NODES_PER_PASS = 100_000

ExactCoverSearchStatus = Literal["FOUND", "NO_COVER", "UNKNOWN"]
MinimumExactCoverStatus = Literal["EXACT", "INFEASIBLE", "BOUNDED"]


def _require_canonical_labels(labels: tuple[str, ...], role: str) -> None:
    if any(not unicodedata.is_normalized("NFC", label) for label in labels):
        raise _combinatorics_validation_error(f"{role} must use Unicode NFC")
    if labels != tuple(sorted(set(labels))):
        raise _combinatorics_validation_error(f"{role} must be sorted and unique")


class ExactCoverRow(StrictModel):
    """One identified finite row in a generalized exact-cover instance."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "One materialized incidence row. `row_id` is an opaque unique "
                "identifier; `items` must be sorted, unique Unicode-NFC labels. "
                "Every item is declared by the enclosing instance. Duplicate "
                "incidences are rejected rather than normalized."
            )
        }
    )

    row_id: OpaqueLabel
    items: tuple[OpaqueLabel, ...] = Field(max_length=MAX_EXACT_COVER_ITEMS)

    @model_validator(mode="after")
    def require_canonical_row(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.row_id):
            raise _combinatorics_validation_error("row IDs must use Unicode NFC")
        _require_canonical_labels(self.items, "row items")
        return self


class GeneralizedExactCoverInstance(StrictModel):
    """A materialized finite primary/secondary exact-cover instance.

    A selected row family must cover every primary item exactly once and every
    secondary item at most once. Rows that contain no primary item are valid
    incidence data but are irrelevant to feasibility and are never selected by
    the canonical search.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Canonical materialized generalized exact cover. Primary and "
                "secondary item labels are disjoint, sorted, unique, and use "
                "Unicode NFC. Rows are sorted by unique row ID; each row's item "
                "labels are sorted and unique and must be declared. Rows with "
                "different IDs remain distinct candidates even when their item "
                "sets agree. At most "
                f"{MAX_EXACT_COVER_ITEMS} total items, {MAX_EXACT_COVER_ROWS} "
                f"rows, and {MAX_EXACT_COVER_INCIDENCES} incidences are admitted."
            )
        }
    )

    primary_items: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_EXACT_COVER_PRIMARY_ITEMS,
        description=(
            "Items that a solution covers exactly once, in sorted unique "
            "Unicode-NFC order. The empty tuple is allowed."
        ),
    )
    secondary_items: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_EXACT_COVER_SECONDARY_ITEMS,
        description=(
            "Items that a solution covers at most once, in sorted unique "
            "Unicode-NFC order."
        ),
    )
    rows: tuple[ExactCoverRow, ...] = Field(
        max_length=MAX_EXACT_COVER_ROWS,
        description="Materialized rows in increasing row-ID order.",
    )

    @model_validator(mode="after")
    def require_canonical_instance(self) -> Self:
        _require_canonical_labels(self.primary_items, "primary items")
        _require_canonical_labels(self.secondary_items, "secondary items")
        if set(self.primary_items) & set(self.secondary_items):
            raise _combinatorics_validation_error(
                "primary and secondary items must be disjoint"
            )
        if len(self.primary_items) + len(self.secondary_items) > MAX_EXACT_COVER_ITEMS:
            raise _combinatorics_validation_error(
                f"an exact-cover instance has at most {MAX_EXACT_COVER_ITEMS} items"
            )

        row_ids = tuple(row.row_id for row in self.rows)
        _require_canonical_labels(row_ids, "row IDs")
        declared = set(self.primary_items) | set(self.secondary_items)
        incidence_count = 0
        for row in self.rows:
            if not set(row.items) <= declared:
                raise _combinatorics_validation_error(
                    "every row item must be declared by the instance"
                )
            incidence_count += len(row.items)
        if incidence_count > MAX_EXACT_COVER_INCIDENCES:
            raise _combinatorics_validation_error(
                "exact-cover incidence count exceeds the "
                f"{MAX_EXACT_COVER_INCIDENCES}-incidence bound"
            )
        return self


def exact_cover_instance_digest(instance: GeneralizedExactCoverInstance) -> str:
    payload = instance.model_dump(mode="json")
    return "sha256:" + sha256(canonicalize_json(payload)).hexdigest()


class GeneralizedExactCoverShard(StrictModel):
    """One semantic fixed-row prefix in the canonical exact-cover traversal."""

    instance_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    operation_id: Literal["combinatorics.generalized_exact_cover.find"] = (
        "combinatorics.generalized_exact_cover.find"
    )
    traversal_version: Literal[1] = 1
    fixed_row_prefix: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_EXACT_COVER_PRIMARY_ITEMS
    )


class GeneralizedExactCoverRequest(StrictModel):
    """Find one generalized exact cover under a deterministic node limit."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Find one row family covering each primary item exactly once "
                "and each secondary item at most once. A node-limit stop returns "
                "UNKNOWN with canonical unresolved frontier shards for deterministic "
                "continuation; NO_COVER requires exhaustive completion."
            )
        }
    )

    instance: GeneralizedExactCoverInstance
    shard: GeneralizedExactCoverShard | None = None
    search_node_limit: StrictInt = Field(
        default=100_000,
        ge=1,
        le=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
        description=(
            "Maximum partial selected-row families visited in one deterministic "
            "search pass, including the root and terminal states."
        ),
    )


class ExactCoverItemMultiplicity(StrictModel):
    """One item's reconstructed multiplicity in a selected-row family."""

    item_id: OpaqueLabel
    kind: Literal["PRIMARY", "SECONDARY"]
    multiplicity: StrictInt = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_canonical_item_id(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.item_id):
            raise _combinatorics_validation_error(
                "item multiplicity IDs must use Unicode NFC"
            )
        return self


def _expected_coverage(
    instance: GeneralizedExactCoverInstance,
    selected_row_ids: tuple[str, ...],
) -> tuple[ExactCoverItemMultiplicity, ...]:
    if selected_row_ids != tuple(sorted(set(selected_row_ids))):
        raise _combinatorics_validation_error(
            "selected row IDs must be sorted and unique"
        )
    rows_by_id = {row.row_id: row for row in instance.rows}
    if any(row_id not in rows_by_id for row_id in selected_row_ids):
        raise _combinatorics_validation_error(
            "every selected row ID must be declared by the instance"
        )

    primary = set(instance.primary_items)
    counts = dict.fromkeys((*instance.primary_items, *instance.secondary_items), 0)
    for row_id in selected_row_ids:
        row = rows_by_id[row_id]
        if not primary.intersection(row.items):
            raise _combinatorics_validation_error(
                "a canonical witness omits rows with no primary item"
            )
        for item in row.items:
            counts[item] += 1

    if any(counts[item] != 1 for item in instance.primary_items):
        raise _combinatorics_validation_error(
            "a FOUND witness must cover every primary item exactly once"
        )
    if any(counts[item] > 1 for item in instance.secondary_items):
        raise _combinatorics_validation_error(
            "a FOUND witness must cover every secondary item at most once"
        )

    return tuple(
        ExactCoverItemMultiplicity(
            item_id=item,
            kind="PRIMARY",
            multiplicity=counts[item],
        )
        for item in instance.primary_items
    ) + tuple(
        ExactCoverItemMultiplicity(
            item_id=item,
            kind="SECONDARY",
            multiplicity=counts[item],
        )
        for item in instance.secondary_items
    )


class GeneralizedExactCoverResult(StrictModel):
    """One checked cover, exact nonexistence, or an honest non-conclusion."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Source-bound generalized exact-cover result. FOUND carries a "
                "canonical selected-row family and every declared item's "
                "reconstructed multiplicity. NO_COVER is accepted only after "
                "the producer has exhausted the admitted search. UNKNOWN records "
                "only that this execution made no mathematical conclusion within "
                "the retained node limit."
            )
        }
    )

    instance: GeneralizedExactCoverInstance
    search_node_limit: StrictInt = Field(ge=1, le=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS)
    status: ExactCoverSearchStatus
    source_shard: GeneralizedExactCoverShard | None = None
    selected_row_ids: tuple[OpaqueLabel, ...] | None = Field(
        default=None,
        max_length=MAX_EXACT_COVER_PRIMARY_ITEMS,
        description=(
            "Canonical selected row IDs for FOUND; absent for NO_COVER and UNKNOWN."
        ),
    )
    item_multiplicities: tuple[ExactCoverItemMultiplicity, ...] | None = Field(
        default=None,
        max_length=MAX_EXACT_COVER_ITEMS,
        description=(
            "One reconstructed multiplicity for every declared item in primary-"
            "then-secondary order for FOUND; absent otherwise."
        ),
    )
    searched_node_count: StrictInt = Field(default=1, ge=1, le=2_147_483_647)
    unresolved_frontier: tuple[GeneralizedExactCoverShard, ...] = Field(
        default=(), max_length=MAX_EXACT_COVER_INCIDENCES
    )

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        expected_digest = exact_cover_instance_digest(self.instance)
        if (
            self.source_shard is not None
            and self.source_shard.instance_digest != expected_digest
        ):
            raise _combinatorics_validation_error(
                "source shard must bind the retained exact-cover instance digest"
            )
        if self.status == "FOUND":
            if self.selected_row_ids is None or self.item_multiplicities is None:
                raise _combinatorics_validation_error(
                    "a FOUND result must carry selected rows and item multiplicities"
                )
            if self.unresolved_frontier:
                raise _combinatorics_validation_error(
                    "a FOUND result cannot retain an unresolved frontier"
                )
            return self

        if self.selected_row_ids is not None or self.item_multiplicities is not None:
            raise _combinatorics_validation_error(
                "only a FOUND result may carry a selected-row family"
            )

        if self.status == "UNKNOWN":
            if not self.unresolved_frontier:
                raise _combinatorics_validation_error(
                    "an UNKNOWN result must retain its unresolved frontier"
                )
            if any(
                shard.instance_digest != expected_digest
                for shard in self.unresolved_frontier
            ):
                raise _combinatorics_validation_error(
                    "every unresolved shard must bind the retained instance digest"
                )
        elif self.unresolved_frontier:
            raise _combinatorics_validation_error(
                "an exhaustive result cannot retain unresolved shards"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        instance: GeneralizedExactCoverInstance,
        search_node_limit: int,
        status: ExactCoverSearchStatus,
        selected_row_ids: tuple[OpaqueLabel, ...] | None = None,
        item_multiplicities: tuple[ExactCoverItemMultiplicity, ...] | None = None,
        searched_node_count: int,
        unresolved_frontier: tuple[GeneralizedExactCoverShard, ...] = (),
        source_shard: GeneralizedExactCoverShard | None = None,
    ) -> Self:
        """Construct an admitted, kernel-established result without replay."""

        return cls.model_construct(
            instance=instance,
            search_node_limit=search_node_limit,
            status=status,
            source_shard=source_shard,
            selected_row_ids=selected_row_ids,
            item_multiplicities=item_multiplicities,
            searched_node_count=searched_node_count,
            unresolved_frontier=unresolved_frontier,
        )


class MinimumGeneralizedExactCoverRequest(StrictModel):
    """Minimize the number of selected rows in a generalized exact cover."""

    instance: GeneralizedExactCoverInstance
    search_node_limit: StrictInt = Field(
        default=100_000, ge=1, le=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS
    )


class MinimumGeneralizedExactCoverResult(StrictModel):
    """Exact minimum, exhaustive infeasibility, or witness-backed bounds."""

    instance: GeneralizedExactCoverInstance
    status: MinimumExactCoverStatus
    selected_row_ids: tuple[OpaqueLabel, ...] | None = Field(
        default=None, max_length=MAX_EXACT_COVER_PRIMARY_ITEMS
    )
    item_multiplicities: tuple[ExactCoverItemMultiplicity, ...] | None = Field(
        default=None, max_length=MAX_EXACT_COVER_ITEMS
    )
    lower_bound: StrictInt = Field(ge=0, le=MAX_EXACT_COVER_PRIMARY_ITEMS)
    upper_bound: StrictInt | None = Field(
        default=None, ge=0, le=MAX_EXACT_COVER_PRIMARY_ITEMS
    )
    searched_node_count: StrictInt = Field(
        ge=1, le=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS
    )

    @model_validator(mode="after")
    def bind_minimum_result(self) -> Self:
        if self.status == "INFEASIBLE":
            if (
                self.selected_row_ids is not None
                or self.item_multiplicities is not None
                or self.upper_bound is not None
            ):
                raise _combinatorics_validation_error(
                    "an infeasible minimum result cannot carry an upper bound witness"
                )
            return self
        if (
            self.selected_row_ids is None
            or self.item_multiplicities is None
            or self.upper_bound is None
        ):
            raise _combinatorics_validation_error(
                "an exact or bounded minimum result must carry an upper bound witness"
            )
        expected = _expected_coverage(self.instance, self.selected_row_ids)
        if expected != self.item_multiplicities:
            raise _combinatorics_validation_error(
                "minimum exact-cover multiplicities must reconstruct the witness"
            )
        if self.upper_bound != len(self.selected_row_ids):
            raise _combinatorics_validation_error(
                "minimum exact-cover upper bound must equal its witness cardinality"
            )
        if not self.lower_bound <= self.upper_bound:
            raise _combinatorics_validation_error(
                "minimum exact-cover bounds must be ordered"
            )
        if self.status == "EXACT" and self.lower_bound != self.upper_bound:
            raise _combinatorics_validation_error(
                "an exact minimum must have coincident objective bounds"
            )
        return self


class GeneralizedExactCoverShardSplitRequest(StrictModel):
    instance: GeneralizedExactCoverInstance
    shard: GeneralizedExactCoverShard


class GeneralizedExactCoverShardSplitResult(StrictModel):
    source: GeneralizedExactCoverShardSplitRequest
    children: tuple[GeneralizedExactCoverShard, ...] = Field(
        max_length=MAX_EXACT_COVER_ROWS
    )
    exhausted: bool


class GeneralizedExactCoverShardResultsCombineRequest(StrictModel):
    instance: GeneralizedExactCoverInstance
    parent_shard: GeneralizedExactCoverShard
    child_results: tuple[GeneralizedExactCoverResult, ...] = Field(
        min_length=1, max_length=MAX_EXACT_COVER_ROWS
    )

    @model_validator(mode="after")
    def admit_rechecks(self) -> Self:
        if (
            sum(
                min(result.search_node_limit, result.searched_node_count)
                for result in self.child_results
            )
            > MAX_EXACT_COVER_SEARCH_NODES_PER_PASS
        ):
            raise _combinatorics_validation_error(
                "combined child rechecks exceed the exact-cover node bound"
            )
        return self


def _solve_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    search_node_limit: int,
    shard: GeneralizedExactCoverShard | None = None,
) -> GeneralizedExactCoverResult:
    """Run the kernel after its owner has admitted canonical inputs."""

    from jacobian.math.combinatorics._exact_cover_kernel import (
        search_generalized_exact_cover,
    )

    digest = exact_cover_instance_digest(instance)
    if shard is not None and shard.instance_digest != digest:
        raise _combinatorics_validation_error(
            "exact-cover shard digest must match the canonical instance"
        )
    rows_by_id = {row.row_id: index for index, row in enumerate(instance.rows)}
    try:
        fixed_rows = (
            tuple(rows_by_id[row_id] for row_id in shard.fixed_row_prefix)
            if shard is not None
            else ()
        )
    except KeyError as error:
        raise _combinatorics_validation_error(
            "exact-cover shard fixed rows must belong to the canonical instance"
        ) from error
    try:
        search = search_generalized_exact_cover(instance, search_node_limit, fixed_rows)
    except ValueError as error:
        raise _combinatorics_validation_error(str(error)) from error
    if search.status != "FOUND":
        return GeneralizedExactCoverResult._from_kernel(
            instance=instance,
            search_node_limit=search_node_limit,
            status=search.status,
            source_shard=shard,
            searched_node_count=search.visited_nodes,
            unresolved_frontier=tuple(
                GeneralizedExactCoverShard(
                    instance_digest=digest,
                    fixed_row_prefix=tuple(
                        instance.rows[index].row_id for index in prefix
                    ),
                )
                for prefix in search.frontier_prefixes
            ),
        )

    selected_row_ids = tuple(
        sorted(instance.rows[index].row_id for index in search.selected_rows)
    )
    return GeneralizedExactCoverResult._from_kernel(
        instance=instance,
        search_node_limit=search_node_limit,
        status="FOUND",
        source_shard=shard,
        selected_row_ids=selected_row_ids,
        item_multiplicities=_expected_coverage(instance, selected_row_ids),
        searched_node_count=search.visited_nodes,
    )


def verify_generalized_exact_cover(claim: GeneralizedExactCoverResult) -> bool:
    """Check a FOUND claim's coverage relation against its retained instance.

    Verifies selected-row binding, primary exact coverage, secondary
    at-most-once coverage, and the reconstructed multiplicity profile
    without rerunning the search. Only FOUND claims are verifiable:
    NO_COVER and UNKNOWN are producer outcomes, not reusable claims, so
    they do not verify.
    """
    if claim.status != "FOUND":
        return False
    if claim.selected_row_ids is None or claim.item_multiplicities is None:
        return False
    try:
        expected = _expected_coverage(claim.instance, claim.selected_row_ids)
    except PydanticCustomError:
        return False
    return tuple(claim.item_multiplicities) == expected


def find_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    *,
    search_node_limit: int = MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
    shard: GeneralizedExactCoverShard | None = None,
) -> GeneralizedExactCoverResult:
    """Return one cover, exact nonexistence, or UNKNOWN for canonical values.

    This is the native Python boundary. Catalog and MCP request projection
    remains private to the publication module.
    """

    if not isinstance(instance, GeneralizedExactCoverInstance):
        raise TypeError("instance must be a GeneralizedExactCoverInstance")
    if type(search_node_limit) is not int:
        raise TypeError("search_node_limit must be an integer")
    if not 1 <= search_node_limit <= MAX_EXACT_COVER_SEARCH_NODES_PER_PASS:
        raise _combinatorics_validation_error(
            "search_node_limit must be between 1 and "
            f"{MAX_EXACT_COVER_SEARCH_NODES_PER_PASS}"
        )
    # Price the widest item scan per visited node. Preserve the former
    # 256 items * 100000 nodes * 64 words envelope while allowing deeper,
    # cheap searches. Index and DFS storage stay below 134 million bits.
    work = (
        search_node_limit
        * len(instance.primary_items)
        * max(1, (max(len(instance.rows), len(instance.primary_items)) + 63) // 64)
    )
    if work > 256 * 100_000 * 64:
        from jacobian.catalog.models import OperationDomainValidationError

        raise OperationDomainValidationError(
            location=("search_node_limit",),
            code="combinatorics.exact_cover_work",
            message="item-scan word work exceeds the exact-cover envelope",
        )
    return _solve_generalized_exact_cover(instance, search_node_limit, shard)


def split_generalized_exact_cover_shard(
    request: GeneralizedExactCoverShardSplitRequest,
) -> GeneralizedExactCoverShardSplitResult:
    from jacobian.math.combinatorics._exact_cover_kernel import (
        split_exact_cover_prefix,
    )

    digest = exact_cover_instance_digest(request.instance)
    if request.shard.instance_digest != digest:
        raise _combinatorics_validation_error(
            "exact-cover shard digest must match the canonical instance"
        )
    rows_by_id = {row.row_id: index for index, row in enumerate(request.instance.rows)}
    try:
        prefix = tuple(rows_by_id[row_id] for row_id in request.shard.fixed_row_prefix)
        children = split_exact_cover_prefix(request.instance, prefix)
    except (KeyError, ValueError) as error:
        raise _combinatorics_validation_error(str(error)) from error
    shards = tuple(
        GeneralizedExactCoverShard(
            instance_digest=digest,
            fixed_row_prefix=tuple(
                request.instance.rows[index].row_id for index in child
            ),
        )
        for child in children
    )
    return GeneralizedExactCoverShardSplitResult(
        source=request, children=shards, exhausted=not shards
    )


def combine_generalized_exact_cover_shard_results(
    request: GeneralizedExactCoverShardResultsCombineRequest,
) -> GeneralizedExactCoverResult:
    split = split_generalized_exact_cover_shard(
        GeneralizedExactCoverShardSplitRequest(
            instance=request.instance, shard=request.parent_shard
        )
    )
    expected = {shard.model_dump_json() for shard in split.children}
    actual = {
        result.source_shard.model_dump_json()
        for result in request.child_results
        if result.source_shard is not None
    }
    if actual != expected or len(actual) != len(request.child_results):
        raise _combinatorics_validation_error(
            "combined results must cover each disjoint child shard exactly once"
        )
    confirmed = tuple(
        _solve_generalized_exact_cover(
            request.instance,
            min(result.search_node_limit, result.searched_node_count),
            result.source_shard,
        )
        for result in request.child_results
    )
    searched = sum(result.searched_node_count for result in confirmed)
    found = next(
        (result for result in confirmed if result.status == "FOUND"),
        None,
    )
    if found is not None:
        return GeneralizedExactCoverResult._from_kernel(
            instance=request.instance,
            search_node_limit=max(result.search_node_limit for result in confirmed),
            status="FOUND",
            source_shard=request.parent_shard,
            selected_row_ids=found.selected_row_ids,
            item_multiplicities=found.item_multiplicities,
            searched_node_count=searched,
        )
    if all(result.status == "NO_COVER" for result in confirmed):
        return GeneralizedExactCoverResult._from_kernel(
            instance=request.instance,
            search_node_limit=max(result.search_node_limit for result in confirmed),
            status="NO_COVER",
            source_shard=request.parent_shard,
            searched_node_count=searched,
        )
    frontier = tuple(
        shard for result in confirmed for shard in result.unresolved_frontier
    )
    return GeneralizedExactCoverResult._from_kernel(
        instance=request.instance,
        search_node_limit=max(result.search_node_limit for result in confirmed),
        status="UNKNOWN",
        source_shard=request.parent_shard,
        searched_node_count=searched,
        unresolved_frontier=frontier,
    )


def minimum_generalized_exact_cover(  # noqa: C901
    instance: GeneralizedExactCoverInstance,
    *,
    search_node_limit: int = MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
) -> MinimumGeneralizedExactCoverResult:
    """Minimize selected-row cardinality with honest exhaustive or bounded output."""

    from jacobian.catalog.models import OperationResourceAdmissionError

    if not isinstance(instance, GeneralizedExactCoverInstance):
        raise TypeError("instance must be a GeneralizedExactCoverInstance")
    if type(search_node_limit) is not int or not (
        1 <= search_node_limit <= MAX_EXACT_COVER_SEARCH_NODES_PER_PASS
    ):
        raise _combinatorics_validation_error(
            "search_node_limit must be within the exact-cover node bound"
        )
    items = (*instance.primary_items, *instance.secondary_items)
    item_index = {item: index for index, item in enumerate(items)}
    primary_count = len(instance.primary_items)
    row_count = len(instance.rows)
    scan_work = (
        search_node_limit
        * primary_count
        * max(1, (max(row_count, primary_count) + 63) // 64)
    )
    if scan_work > 256 * 100_000 * 64:
        raise OperationResourceAdmissionError(
            location=("search_node_limit",),
            code="combinatorics.minimum_exact_cover_work",
            message="node-by-item scan work exceeds the minimum exact-cover envelope",
        )
    item_rows = [0] * len(items)
    row_primary_masks: list[int] = []
    row_item_indices: list[tuple[int, ...]] = []
    maximum_primary_coverage = 0
    for row_index, row in enumerate(instance.rows):
        indices = tuple(item_index[item] for item in row.items)
        row_item_indices.append(indices)
        primary_mask = 0
        for index in indices:
            item_rows[index] |= 1 << row_index
            if index < primary_count:
                primary_mask |= 1 << index
        row_primary_masks.append(primary_mask)
        maximum_primary_coverage = max(
            maximum_primary_coverage, primary_mask.bit_count()
        )
    row_conflicts = []
    for indices in row_item_indices:
        conflicts = 0
        for index in indices:
            conflicts |= item_rows[index]
        row_conflicts.append(conflicts)
    all_primary = (1 << primary_count) - 1
    all_rows = (1 << row_count) - 1
    root_lower_bound = (
        (primary_count + maximum_primary_coverage - 1) // maximum_primary_coverage
        if primary_count and maximum_primary_coverage
        else 0
    )
    stack: list[tuple[int, int, tuple[int, ...]]] = [(all_primary, all_rows, ())]
    incumbent: tuple[int, ...] | None = None
    incumbent_ids: tuple[str, ...] | None = None
    visited = 0
    while stack and visited < search_node_limit:
        uncovered, available, selected = stack.pop()
        visited += 1
        if uncovered == 0:
            selected_ids = tuple(
                sorted(instance.rows[index].row_id for index in selected)
            )
            if (
                incumbent is None
                or len(selected) < len(incumbent)
                or (
                    len(selected) == len(incumbent)
                    and (incumbent_ids is None or selected_ids < incumbent_ids)
                )
            ):
                incumbent = selected
                incumbent_ids = selected_ids
            continue
        if incumbent is not None and len(selected) >= len(incumbent) - 1:
            continue
        chosen_rows = 0
        fewest = row_count + 1
        remaining = uncovered
        while remaining:
            bit = remaining & -remaining
            item = bit.bit_length() - 1
            candidates = item_rows[item] & available
            if candidates.bit_count() < fewest:
                fewest = candidates.bit_count()
                chosen_rows = candidates
            remaining ^= bit
        candidate_indices: list[int] = []
        while chosen_rows:
            bit = chosen_rows & -chosen_rows
            candidate_indices.append(bit.bit_length() - 1)
            chosen_rows ^= bit
        for row_index in reversed(candidate_indices):
            stack.append(
                (
                    uncovered & ~row_primary_masks[row_index],
                    available & ~row_conflicts[row_index],
                    (*selected, row_index),
                )
            )
    exhausted = not stack
    if incumbent is None:
        if not exhausted:
            raise OperationResourceAdmissionError(
                location=("search_node_limit",),
                code="combinatorics.minimum_exact_cover.no_incumbent",
                message=(
                    "node limit was reached before finding a feasible upper-bound "
                    "witness; increase search_node_limit"
                ),
            )
        return MinimumGeneralizedExactCoverResult(
            instance=instance,
            status="INFEASIBLE",
            lower_bound=root_lower_bound,
            searched_node_count=visited,
        )
    assert incumbent_ids is not None
    selected_ids = incumbent_ids
    upper_bound = len(selected_ids)
    return MinimumGeneralizedExactCoverResult(
        instance=instance,
        status="EXACT" if exhausted else "BOUNDED",
        selected_row_ids=selected_ids,
        item_multiplicities=_expected_coverage(instance, selected_ids),
        lower_bound=upper_bound if exhausted else min(root_lower_bound, upper_bound),
        upper_bound=upper_bound,
        searched_node_count=visited,
    )


__all__ = [
    "ExactCoverItemMultiplicity",
    "ExactCoverRow",
    "ExactCoverSearchStatus",
    "GeneralizedExactCoverInstance",
    "GeneralizedExactCoverResult",
    "GeneralizedExactCoverShard",
    "GeneralizedExactCoverShardResultsCombineRequest",
    "GeneralizedExactCoverShardSplitRequest",
    "GeneralizedExactCoverShardSplitResult",
    "MinimumExactCoverStatus",
    "MinimumGeneralizedExactCoverRequest",
    "MinimumGeneralizedExactCoverResult",
    "combine_generalized_exact_cover_shard_results",
    "exact_cover_instance_digest",
    "find_generalized_exact_cover",
    "minimum_generalized_exact_cover",
    "split_generalized_exact_cover_shard",
    "verify_generalized_exact_cover",
]
