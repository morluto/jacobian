"""Public declaration for bounded generalized exact cover."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.combinatorics.exact_cover import (
    GeneralizedExactCoverRequest,
    GeneralizedExactCoverResult,
    find_generalized_exact_cover,
)

GENERALIZED_EXACT_COVER_OPERATION = MathTool(
    operation_id="combinatorics.generalized_exact_cover.find",
    version="1",
    title="Find a generalized exact cover",
    description=(
        "Find one row family that covers every primary item exactly once and "
        "every secondary item at most once. Return FOUND with a checked "
        "selected-row family, NO_COVER only after complete bounded search, or "
        "UNKNOWN when the deterministic search-node limit is reached."
    ),
    request_type=GeneralizedExactCoverRequest,
    result_type=GeneralizedExactCoverResult,
    run=find_generalized_exact_cover,
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
        example(
            "two_constraints_one_resource",
            "Select one row for each of two primary constraints while using "
            "the optional resource at most once; item and row labels must be "
            "declared in sorted canonical order.",
            {
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

__all__ = ["GENERALIZED_EXACT_COVER_OPERATION"]
