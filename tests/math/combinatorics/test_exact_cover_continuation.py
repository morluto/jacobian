"""Deterministic generalized exact-cover continuation tests."""

import time
from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_execution,
)
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
