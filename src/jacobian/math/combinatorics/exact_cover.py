"""Canonical values and results for bounded generalized exact cover."""

from __future__ import annotations

import unicodedata
from builtins import ValueError as BuiltinValueError
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    encode_strict_json,
)
from jacobian.math._labels import OpaqueLabel


def _combinatorics_validation_error(message: str) -> PydanticCustomError:
    lowered = message.lower()
    code = "combinatorics.exact_cover_invariant"
    if "bound" in lowered or "count" in lowered or "limit" in lowered:
        code = "combinatorics.exact_cover_bound"
    return PydanticCustomError(code, message, {})


ValueError = _combinatorics_validation_error  # noqa: A001

MAX_EXACT_COVER_ITEMS = 256
MAX_EXACT_COVER_PRIMARY_ITEMS = MAX_EXACT_COVER_ITEMS
MAX_EXACT_COVER_SECONDARY_ITEMS = MAX_EXACT_COVER_ITEMS
MAX_EXACT_COVER_ROWS = 4_096
MAX_EXACT_COVER_INCIDENCES = 65_536

# A million-node adversarial pass is too slow to repeat at the public boundary;
# 100,000 nodes per pass is a measured conservative execution fallback,
# independent of the broader 256-item representation bound.
MAX_EXACT_COVER_SEARCH_NODES_PER_PASS = 100_000

ExactCoverSearchStatus = Literal["FOUND", "NO_COVER", "UNKNOWN"]


def _require_canonical_labels(labels: tuple[str, ...], role: str) -> None:
    if any(not unicodedata.is_normalized("NFC", label) for label in labels):
        raise ValueError(f"{role} must use Unicode NFC")
    if labels != tuple(sorted(set(labels))):
        raise ValueError(f"{role} must be sorted and unique")


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
            raise ValueError("row IDs must use Unicode NFC")
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
            raise ValueError("primary and secondary items must be disjoint")
        if len(self.primary_items) + len(self.secondary_items) > MAX_EXACT_COVER_ITEMS:
            raise ValueError(
                f"an exact-cover instance has at most {MAX_EXACT_COVER_ITEMS} items"
            )

        row_ids = tuple(row.row_id for row in self.rows)
        _require_canonical_labels(row_ids, "row IDs")
        declared = set(self.primary_items) | set(self.secondary_items)
        incidence_count = 0
        for row in self.rows:
            if not set(row.items) <= declared:
                raise ValueError("every row item must be declared by the instance")
            incidence_count += len(row.items)
        if incidence_count > MAX_EXACT_COVER_INCIDENCES:
            raise ValueError(
                "exact-cover incidence count exceeds the "
                f"{MAX_EXACT_COVER_INCIDENCES}-incidence bound"
            )
        return self


def _coverage_wire_value(
    instance: GeneralizedExactCoverInstance,
) -> list[dict[str, object]]:
    return [
        {"item_id": item, "kind": "PRIMARY", "multiplicity": 1}
        for item in instance.primary_items
    ] + [
        {"item_id": item, "kind": "SECONDARY", "multiplicity": 1}
        for item in instance.secondary_items
    ]


def _require_output_headroom(instance: GeneralizedExactCoverInstance) -> None:
    """Prove that every result shape fits before the search begins.

    A witness contains at most one selected row per primary item. The coverage
    ledger contains one bounded record per declared item. The exact source is
    retained once so conclusions remain replayable. Materializing the largest
    possible FOUND wire value and both non-positive wire values gives the exact
    serialized upper bound for this source; it does not rely on label lengths
    or a fixed structural reserve.
    """

    try:
        source = instance.model_dump(mode="json")
        selected_count = min(len(instance.primary_items), len(instance.rows))
        selected_row_ids = sorted(
            instance.rows,
            key=lambda row: len(encode_strict_json(row.row_id)),
            reverse=True,
        )[:selected_count]
        common = {
            "instance": source,
            "search_node_limit": MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
        }
        found_wire = {
            **common,
            "status": "FOUND",
            "selected_row_ids": sorted(row.row_id for row in selected_row_ids),
            "item_multiplicities": _coverage_wire_value(instance),
        }
        no_cover_wire = {
            **common,
            "status": "NO_COVER",
            "selected_row_ids": None,
            "item_multiplicities": None,
        }
        unknown_wire = {
            **common,
            "status": "UNKNOWN",
            "selected_row_ids": None,
            "item_multiplicities": None,
        }
        for wire in (found_wire, no_cover_wire, unknown_wire):
            encode_strict_json(wire)
    except CanonicalizationError as exc:
        raise ValueError(
            "the exact-cover result retains its source and would exceed the "
            "canonical output limit; shorten labels or shrink the incidence data"
        ) from exc


class GeneralizedExactCoverRequest(StrictModel):
    """Find one generalized exact cover under a deterministic node limit."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Find one row family covering each primary item exactly once "
                "and each secondary item at most once. The bitset Algorithm X "
                "search visits at most `search_node_limit` partial row families "
                "per pass. It returns NO_COVER only after exhaustive search; a "
                "node-limit stop returns UNKNOWN. UNKNOWN is a source-bound "
                "non-conclusion rather than a proof of absence."
            )
        }
    )

    instance: GeneralizedExactCoverInstance
    search_node_limit: StrictInt = Field(
        default=100_000,
        ge=1,
        le=MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
        description=(
            "Maximum partial selected-row families visited in one deterministic "
            "search pass, including the root and terminal states."
        ),
    )

    @model_validator(mode="after")
    def reserve_result_space(self) -> Self:
        _require_output_headroom(self.instance)
        return self


class ExactCoverItemMultiplicity(StrictModel):
    """One item's reconstructed multiplicity in a selected-row family."""

    item_id: OpaqueLabel
    kind: Literal["PRIMARY", "SECONDARY"]
    multiplicity: StrictInt = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_canonical_item_id(self) -> Self:
        if not unicodedata.is_normalized("NFC", self.item_id):
            raise ValueError("item multiplicity IDs must use Unicode NFC")
        return self


def _expected_coverage(
    instance: GeneralizedExactCoverInstance,
    selected_row_ids: tuple[str, ...],
) -> tuple[ExactCoverItemMultiplicity, ...]:
    if selected_row_ids != tuple(sorted(set(selected_row_ids))):
        raise ValueError("selected row IDs must be sorted and unique")
    rows_by_id = {row.row_id: row for row in instance.rows}
    if any(row_id not in rows_by_id for row_id in selected_row_ids):
        raise ValueError("every selected row ID must be declared by the instance")

    primary = set(instance.primary_items)
    counts = dict.fromkeys((*instance.primary_items, *instance.secondary_items), 0)
    for row_id in selected_row_ids:
        row = rows_by_id[row_id]
        if not primary.intersection(row.items):
            raise ValueError("a canonical witness omits rows with no primary item")
        for item in row.items:
            counts[item] += 1

    if any(counts[item] != 1 for item in instance.primary_items):
        raise ValueError("a FOUND witness must cover every primary item exactly once")
    if any(counts[item] > 1 for item in instance.secondary_items):
        raise ValueError("a FOUND witness must cover every secondary item at most once")

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

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        if self.status == "FOUND":
            if self.selected_row_ids is None or self.item_multiplicities is None:
                raise ValueError(
                    "a FOUND result must carry selected rows and item multiplicities"
                )
            return self

        if self.selected_row_ids is not None or self.item_multiplicities is not None:
            raise ValueError("only a FOUND result may carry a selected-row family")

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
    ) -> Self:
        """Construct an admitted, kernel-established result without replay."""

        return cls.model_construct(
            instance=instance,
            search_node_limit=search_node_limit,
            status=status,
            selected_row_ids=selected_row_ids,
            item_multiplicities=item_multiplicities,
        )


def _verify_generalized_exact_cover_result(result: GeneralizedExactCoverResult) -> bool:
    """Replay an independently supplied exact-cover claim in its envelope."""

    try:
        request = GeneralizedExactCoverRequest(
            instance=result.instance, search_node_limit=result.search_node_limit
        )
        if result.status == "FOUND":
            if result.selected_row_ids is None or result.item_multiplicities is None:
                return False
            if result.item_multiplicities != _expected_coverage(
                request.instance, result.selected_row_ids
            ):
                return False
        elif (
            result.selected_row_ids is not None
            or result.item_multiplicities is not None
        ):
            return False

        from jacobian.math.combinatorics._exact_cover_kernel import (
            search_generalized_exact_cover,
        )

        replay = search_generalized_exact_cover(
            request.instance, request.search_node_limit
        )
    except (BuiltinValueError, TypeError):
        return False

    if replay.status != result.status:
        return False
    if result.status != "FOUND":
        return True
    assert result.selected_row_ids is not None
    return result.selected_row_ids == tuple(
        sorted(request.instance.rows[index].row_id for index in replay.selected_rows)
    )


def _solve_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    search_node_limit: int,
) -> GeneralizedExactCoverResult:
    """Run the kernel after its owner has admitted canonical inputs."""

    from jacobian.math.combinatorics._exact_cover_kernel import (
        search_generalized_exact_cover,
    )

    search = search_generalized_exact_cover(instance, search_node_limit)
    if search.status != "FOUND":
        return GeneralizedExactCoverResult._from_kernel(
            instance=instance,
            search_node_limit=search_node_limit,
            status=search.status,
        )

    selected_row_ids = tuple(
        sorted(instance.rows[index].row_id for index in search.selected_rows)
    )
    return GeneralizedExactCoverResult._from_kernel(
        instance=instance,
        search_node_limit=search_node_limit,
        status="FOUND",
        selected_row_ids=selected_row_ids,
        item_multiplicities=_expected_coverage(instance, selected_row_ids),
    )


def find_generalized_exact_cover(
    instance: GeneralizedExactCoverInstance,
    *,
    search_node_limit: int = MAX_EXACT_COVER_SEARCH_NODES_PER_PASS,
) -> GeneralizedExactCoverResult:
    """Return one cover, exact nonexistence, or UNKNOWN for canonical values.

    This is the native Python boundary. Catalog and MCP execution retain their
    strict wire request model in :func:`_find_generalized_exact_cover_request`.
    """

    if not isinstance(instance, GeneralizedExactCoverInstance):
        raise TypeError("instance must be a GeneralizedExactCoverInstance")
    if type(search_node_limit) is not int:
        raise TypeError("search_node_limit must be an integer")
    if not 1 <= search_node_limit <= MAX_EXACT_COVER_SEARCH_NODES_PER_PASS:
        raise ValueError(
            "search_node_limit must be between 1 and "
            f"{MAX_EXACT_COVER_SEARCH_NODES_PER_PASS}"
        )
    _require_output_headroom(instance)
    return _solve_generalized_exact_cover(instance, search_node_limit)


def _find_generalized_exact_cover_request(
    request: GeneralizedExactCoverRequest,
) -> GeneralizedExactCoverResult:
    """Catalog adapter for the strict generalized-exact-cover wire request."""

    return _solve_generalized_exact_cover(request.instance, request.search_node_limit)


__all__ = [
    "ExactCoverItemMultiplicity",
    "ExactCoverRow",
    "ExactCoverSearchStatus",
    "GeneralizedExactCoverInstance",
    "GeneralizedExactCoverRequest",
    "GeneralizedExactCoverResult",
    "find_generalized_exact_cover",
]
