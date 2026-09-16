"""Deterministic generalized exact-cover continuation tests."""

import time
from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.exact_cover import (
    ExactCoverRow,
    GeneralizedExactCoverInstance,
    GeneralizedExactCoverShard,
    GeneralizedExactCoverShardResultsCombineRequest,
    GeneralizedExactCoverShardSplitRequest,
    combine_generalized_exact_cover_shard_results,
    exact_cover_instance_digest,
    find_generalized_exact_cover,
    split_generalized_exact_cover_shard,
)


def _root(instance: GeneralizedExactCoverInstance) -> GeneralizedExactCoverShard:
    return GeneralizedExactCoverShard(
        instance_digest=exact_cover_instance_digest(instance), fixed_row_prefix=()
    )


def test_node_limit_returns_canonical_resumable_frontier() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",),
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="a", items=("p",)),
            ExactCoverRow(row_id="b", items=("p",)),
        ),
    )
    limited = find_generalized_exact_cover(instance, search_node_limit=1)
    assert [shard.fixed_row_prefix for shard in limited.unresolved_frontier] == [
        ("a",),
        ("b",),
    ]
    resumed = find_generalized_exact_cover(
        instance, search_node_limit=1, shard=limited.unresolved_frontier[0]
    )
    assert resumed.status == "FOUND"
    assert resumed.selected_row_ids == ("a",)


def test_split_is_disjoint_complete_and_children_combine_negative() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p", "q"),
        secondary_items=("s",),
        rows=(
            ExactCoverRow(row_id="a", items=("p", "s")),
            ExactCoverRow(row_id="b", items=("p", "s")),
            ExactCoverRow(row_id="c", items=("q", "s")),
            ExactCoverRow(row_id="d", items=("q", "s")),
        ),
    )
    root = _root(instance)
    split = split_generalized_exact_cover_shard(
        GeneralizedExactCoverShardSplitRequest(instance=instance, shard=root)
    )
    assert [child.fixed_row_prefix for child in split.children] == [("a",), ("b",)]
    child_results = tuple(
        find_generalized_exact_cover(instance, shard=child) for child in split.children
    )
    assert all(result.status == "NO_COVER" for result in child_results)
    combined = combine_generalized_exact_cover_shard_results(
        GeneralizedExactCoverShardResultsCombineRequest(
            instance=instance, parent_shard=root, child_results=child_results
        )
    )
    assert combined.status == "NO_COVER"
    assert combined.source_shard == root


def test_combine_rechecks_caller_authored_negative_results() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",),
        secondary_items=(),
        rows=(ExactCoverRow(row_id="a", items=("p",)),),
    )
    root = _root(instance)
    child = split_generalized_exact_cover_shard(
        GeneralizedExactCoverShardSplitRequest(instance=instance, shard=root)
    ).children[0]
    actual = find_generalized_exact_cover(instance, shard=child)
    forged = actual.model_copy(
        update={
            "status": "NO_COVER",
            "selected_row_ids": None,
            "item_multiplicities": None,
        }
    )
    combined = combine_generalized_exact_cover_shard_results(
        GeneralizedExactCoverShardResultsCombineRequest(
            instance=instance, parent_shard=root, child_results=(forged,)
        )
    )
    assert combined.status == "FOUND"
    assert combined.selected_row_ids == ("a",)


def test_exact_cover_search_observes_cancellation_inside_the_kernel() -> None:
    """A cancelled request must stop during node expansion, not after the search."""
    instance = GeneralizedExactCoverInstance(
        primary_items=("p",),
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="a", items=("p",)),
            ExactCoverRow(row_id="b", items=("p",)),
        ),
    )
    cancellation = Event()
    cancellation.set()
    with (
        request_execution(time.monotonic()),
        request_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="exact-cover search"),
    ):
        find_generalized_exact_cover(instance, search_node_limit=10_000)


# ---------------------------------------------------------------------------
# Enumeration laws: exact-cover partitions (Workstream C)
# ---------------------------------------------------------------------------


def _primary_blocks(
    instance: GeneralizedExactCoverInstance, selected_row_ids: tuple[str, ...]
) -> list[frozenset[str]]:
    rows_by_id = {row.row_id: row for row in instance.rows}
    primary = set(instance.primary_items)
    return [
        frozenset(item for item in rows_by_id[row_id].items if item in primary)
        for row_id in selected_row_ids
    ]


def _assert_exact_partition(
    blocks: list[frozenset[str]], universe: tuple[str, ...]
) -> None:
    for block in blocks:
        assert block, "partition blocks must be nonempty"
    seen: set[str] = set()
    for block in blocks:
        assert not seen.intersection(block), "partition blocks must be disjoint"
        seen.update(block)
    assert seen == set(universe), "partition blocks must cover the universe"


def test_found_cover_partitions_the_primary_universe() -> None:
    from jacobian.math.combinatorics.exact_cover import verify_generalized_exact_cover

    instance = GeneralizedExactCoverInstance(
        primary_items=("p", "q", "r"),
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="a", items=("p",)),
            ExactCoverRow(row_id="b", items=("q",)),
            ExactCoverRow(row_id="c", items=("r",)),
            ExactCoverRow(row_id="d", items=("p", "q")),
        ),
    )
    result = find_generalized_exact_cover(instance)
    assert result.status == "FOUND"
    assert result.selected_row_ids is not None
    assert verify_generalized_exact_cover(result)
    _assert_exact_partition(
        _primary_blocks(instance, result.selected_row_ids), instance.primary_items
    )
    assert result.item_multiplicities is not None
    assert all(
        entry.multiplicity == 1
        for entry in result.item_multiplicities
        if entry.kind == "PRIMARY"
    )


def test_empty_universe_empty_family_is_found_empty() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=(), secondary_items=(), rows=()
    )
    result = find_generalized_exact_cover(instance)
    assert result.status == "FOUND"
    assert result.selected_row_ids == ()
    assert result.selected_row_ids is not None
    _assert_exact_partition(
        _primary_blocks(instance, result.selected_row_ids), instance.primary_items
    )


def test_forged_subfamily_fails_partition_and_verification() -> None:
    from jacobian.math.combinatorics.exact_cover import verify_generalized_exact_cover

    instance = GeneralizedExactCoverInstance(
        primary_items=("p", "q"),
        secondary_items=("s",),
        rows=(
            ExactCoverRow(row_id="a", items=("p", "s")),
            ExactCoverRow(row_id="b", items=("q", "s")),
            ExactCoverRow(row_id="c", items=("p", "q")),
        ),
    )
    genuine = find_generalized_exact_cover(instance)
    assert genuine.status == "FOUND"
    assert genuine.selected_row_ids is not None
    assert verify_generalized_exact_cover(genuine)
    # A proper subfamily with valid shape no longer covers the universe.
    forged = genuine.model_copy(update={"selected_row_ids": ("a",)})
    assert not verify_generalized_exact_cover(forged)
    assert _primary_blocks(instance, ("a",)) == [frozenset({"p"})]
    assert set(instance.primary_items) != frozenset({"p"})
    # An overlapping secondary pair keeps valid primary blocks but covers s twice.
    double = genuine.model_copy(update={"selected_row_ids": ("a", "b")})
    assert not verify_generalized_exact_cover(double)


def test_combine_rejects_partial_child_coverage() -> None:
    instance = GeneralizedExactCoverInstance(
        primary_items=("p", "q"),
        secondary_items=("s",),
        rows=(
            ExactCoverRow(row_id="a", items=("p", "s")),
            ExactCoverRow(row_id="b", items=("p", "s")),
            ExactCoverRow(row_id="c", items=("q", "s")),
            ExactCoverRow(row_id="d", items=("q", "s")),
        ),
    )
    root = _root(instance)
    children = split_generalized_exact_cover_shard(
        GeneralizedExactCoverShardSplitRequest(instance=instance, shard=root)
    ).children
    assert len(children) == 2
    partial = (find_generalized_exact_cover(instance, shard=children[0]),)
    with pytest.raises(ValueError, match="disjoint"):
        combine_generalized_exact_cover_shard_results(
            GeneralizedExactCoverShardResultsCombineRequest(
                instance=instance, parent_shard=root, child_results=partial
            )
        )


def test_shard_semantic_rejections_are_owner_typed() -> None:
    """Shard digest and coverage failures use the typed domain rejection.

    ``PydanticCustomError`` (a bare ``ValueError``) escaping an operation body
    would bypass the canonical ``OperationDomainValidationError`` boundary and
    surface an untyped error to the caller.
    """

    instance = GeneralizedExactCoverInstance(
        primary_items=("p",),
        secondary_items=(),
        rows=(
            ExactCoverRow(row_id="a", items=("p",)),
            ExactCoverRow(row_id="b", items=("p",)),
        ),
    )
    stale = GeneralizedExactCoverShard(
        instance_digest="sha256:" + "1" * 64, fixed_row_prefix=()
    )

    with pytest.raises(OperationDomainValidationError) as split_error:
        split_generalized_exact_cover_shard(
            GeneralizedExactCoverShardSplitRequest(instance=instance, shard=stale)
        )
    assert split_error.value.errors()[0]["loc"] == ("shard", "instance_digest")

    root = _root(instance)
    children = split_generalized_exact_cover_shard(
        GeneralizedExactCoverShardSplitRequest(instance=instance, shard=root)
    ).children
    partial = (find_generalized_exact_cover(instance, shard=children[0]),)
    with pytest.raises(OperationDomainValidationError) as combine_error:
        combine_generalized_exact_cover_shard_results(
            GeneralizedExactCoverShardResultsCombineRequest(
                instance=instance, parent_shard=root, child_results=partial
            )
        )
    assert combine_error.value.errors()[0]["loc"] == ("child_results",)
