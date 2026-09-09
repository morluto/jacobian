"""Deterministic generalized exact-cover continuation tests."""

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
