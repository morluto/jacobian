"""Public declaration for bounded generalized exact cover."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.exact_cover import (
    GeneralizedExactCoverRequest,
    GeneralizedExactCoverResult,
    GeneralizedExactCoverShardResultsCombineRequest,
    GeneralizedExactCoverShardSplitRequest,
    GeneralizedExactCoverShardSplitResult,
    MinimumGeneralizedExactCoverRequest,
    MinimumGeneralizedExactCoverResult,
    combine_generalized_exact_cover_shard_results,
    find_generalized_exact_cover,
    minimum_generalized_exact_cover,
    split_generalized_exact_cover_shard,
)


def _run_generalized_exact_cover(
    request: GeneralizedExactCoverRequest,
) -> GeneralizedExactCoverResult:
    return find_generalized_exact_cover(
        request.instance,
        search_node_limit=request.search_node_limit,
        shard=request.shard,
    )


def _run_minimum_generalized_exact_cover(
    request: MinimumGeneralizedExactCoverRequest,
) -> MinimumGeneralizedExactCoverResult:
    return minimum_generalized_exact_cover(
        request.instance, search_node_limit=request.search_node_limit
    )


def _split_shard(
    request: GeneralizedExactCoverShardSplitRequest,
) -> GeneralizedExactCoverShardSplitResult:
    return split_generalized_exact_cover_shard(request)


def _combine_shard_results(
    request: GeneralizedExactCoverShardResultsCombineRequest,
) -> GeneralizedExactCoverResult:
    return combine_generalized_exact_cover_shard_results(request)


GENERALIZED_EXACT_COVER_OPERATION = MathTool(
    operation_id="combinatorics.generalized_exact_cover.find",
    title="Find a generalized exact cover",
    description=(
        "Find one row family that covers every primary item exactly once and "
        "every secondary item at most once. Return FOUND with a checked "
        "selected-row family, NO_COVER only after complete bounded search, or "
        "UNKNOWN when the deterministic search-node limit is reached."
    ),
    request_type=GeneralizedExactCoverRequest,
    result_type=GeneralizedExactCoverResult,
    run=_run_generalized_exact_cover,
    tags=(
        "combinatorics",
        "exact-cover",
        "generalized-exact-cover",
        "primary-items",
        "secondary-items",
        "incidence",
        "algorithm-x",
        "bounded-search",
        "deterministic",
    ),
    examples=(
        OperationExample(
            name="two_constraints_one_resource",
            description="Select one row for each of two primary constraints while using "
            "the optional resource at most once; item and row labels must be "
            "declared in sorted canonical order.",
            input={
                "instance": {
                    "primary_items": ["constraint:a", "constraint:b"],
                    "secondary_items": ["resource:x"],
                    "rows": [
                        {
                            "row_id": "a-use-x",
                            "items": ["constraint:a", "resource:x"],
                        },
                        {
                            "row_id": "b-free",
                            "items": ["constraint:b"],
                        },
                    ],
                },
                "search_node_limit": 100,
            },
        ),
    ),
)

MINIMUM_GENERALIZED_EXACT_COVER_OPERATION = MathTool(
    operation_id="combinatorics.generalized_exact_cover.minimum.compute",
    title="Compute a minimum generalized exact cover",
    description=(
        "Minimize selected-row cardinality. EXACT and INFEASIBLE require exhaustive "
        "search; BOUNDED carries an attaining cover and rigorous objective bounds."
    ),
    request_type=MinimumGeneralizedExactCoverRequest,
    result_type=MinimumGeneralizedExactCoverResult,
    run=_run_minimum_generalized_exact_cover,
    tags=("combinatorics", "exact-cover", "optimization", "bounded", "exact"),
    examples=(
        OperationExample(
            name="one_row_beats_two",
            description=(
                "Cover two primary items with the single combined row; primary "
                "and secondary items and row IDs must be sorted canonical labels."
            ),
            input={
                "instance": {
                    "primary_items": ["p", "q"],
                    "secondary_items": [],
                    "rows": [
                        {"row_id": "both", "items": ["p", "q"]},
                        {"row_id": "p-only", "items": ["p"]},
                        {"row_id": "q-only", "items": ["q"]},
                    ],
                },
                "search_node_limit": 100,
            },
        ),
    ),
)

GENERALIZED_EXACT_COVER_SHARD_SPLIT_OPERATION = MathTool(
    operation_id="combinatorics.generalized_exact_cover.shard.split",
    title="Split a generalized exact-cover search shard",
    description="Split one canonical semantic fixed-row prefix into a disjoint complete family of child search shards.",
    request_type=GeneralizedExactCoverShardSplitRequest,
    result_type=GeneralizedExactCoverShardSplitResult,
    run=_split_shard,
    tags=("combinatorics", "exact-cover", "continuation", "shard"),
    examples=(
        OperationExample(
            name="split_root",
            description="Split the canonical root of a two-choice exact-cover search.",
            input={
                "instance": {
                    "primary_items": ["p"],
                    "secondary_items": [],
                    "rows": [
                        {"row_id": "a", "items": ["p"]},
                        {"row_id": "b", "items": ["p"]},
                    ],
                },
                "shard": {
                    "instance_digest": "sha256:5eced491c2223f527986c8483f555250fe68a3ab3548d1daca8bd582a628b6ef",
                    "fixed_row_prefix": [],
                },
            },
        ),
    ),
)

GENERALIZED_EXACT_COVER_SHARD_COMBINE_OPERATION = MathTool(
    operation_id="combinatorics.generalized_exact_cover.shard_results.combine",
    title="Combine generalized exact-cover shard results",
    description="Combine a complete set of disjoint child results: any cover wins, absence requires every child exhausted, and otherwise unresolved children remain.",
    request_type=GeneralizedExactCoverShardResultsCombineRequest,
    result_type=GeneralizedExactCoverResult,
    run=_combine_shard_results,
    tags=("combinatorics", "exact-cover", "continuation", "combine"),
    examples=(
        OperationExample(
            name="combine_two_exhausted_children",
            description="Combine exhaustive negative results for both children of a root shard.",
            input={
                "instance": {
                    "primary_items": ["p", "q"],
                    "secondary_items": ["s"],
                    "rows": [
                        {"row_id": "a", "items": ["p", "s"]},
                        {"row_id": "b", "items": ["p", "s"]},
                        {"row_id": "c", "items": ["q", "s"]},
                        {"row_id": "d", "items": ["q", "s"]},
                    ],
                },
                "parent_shard": {
                    "instance_digest": "sha256:5c941b1b67b251ebcbfb27c1b31bdb2617d9406d6ce76d28e9b51df4d946e0eb",
                    "fixed_row_prefix": [],
                },
                "child_results": [
                    {
                        "instance": {
                            "primary_items": ["p", "q"],
                            "secondary_items": ["s"],
                            "rows": [
                                {"row_id": "a", "items": ["p", "s"]},
                                {"row_id": "b", "items": ["p", "s"]},
                                {"row_id": "c", "items": ["q", "s"]},
                                {"row_id": "d", "items": ["q", "s"]},
                            ],
                        },
                        "search_node_limit": 1,
                        "status": "NO_COVER",
                        "source_shard": {
                            "instance_digest": "sha256:5c941b1b67b251ebcbfb27c1b31bdb2617d9406d6ce76d28e9b51df4d946e0eb",
                            "fixed_row_prefix": ["a"],
                        },
                        "searched_node_count": 1,
                    },
                    {
                        "instance": {
                            "primary_items": ["p", "q"],
                            "secondary_items": ["s"],
                            "rows": [
                                {"row_id": "a", "items": ["p", "s"]},
                                {"row_id": "b", "items": ["p", "s"]},
                                {"row_id": "c", "items": ["q", "s"]},
                                {"row_id": "d", "items": ["q", "s"]},
                            ],
                        },
                        "search_node_limit": 1,
                        "status": "NO_COVER",
                        "source_shard": {
                            "instance_digest": "sha256:5c941b1b67b251ebcbfb27c1b31bdb2617d9406d6ce76d28e9b51df4d946e0eb",
                            "fixed_row_prefix": ["b"],
                        },
                        "searched_node_count": 1,
                    },
                ],
            },
        ),
    ),
)

__all__ = [
    "GENERALIZED_EXACT_COVER_OPERATION",
    "GENERALIZED_EXACT_COVER_SHARD_COMBINE_OPERATION",
    "GENERALIZED_EXACT_COVER_SHARD_SPLIT_OPERATION",
    "MINIMUM_GENERALIZED_EXACT_COVER_OPERATION",
]
