"""Weighted monotone subsequence endpoint profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    EndpointProfileRequest,
    EndpointProfileResult,
    WeightedMaximumRequest,
    WeightedMaximumResult,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone.operations import (
    compute_endpoint_profile,
    maximum_weight_nondecreasing_subsequence,
    maximum_weight_nonincreasing_subsequence,
)


def compute_endpoint_profile_op(
    request: EndpointProfileRequest,
) -> EndpointProfileResult:
    return compute_endpoint_profile(request.source)


def maximum_weight_nondecreasing_subsequence_op(
    request: WeightedMaximumRequest,
) -> WeightedMaximumResult:
    return maximum_weight_nondecreasing_subsequence(request.source)


def maximum_weight_nonincreasing_subsequence_op(
    request: WeightedMaximumRequest,
) -> WeightedMaximumResult:
    return maximum_weight_nonincreasing_subsequence(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="algebraic_combinatorics.weighted_monotone_subsequence.endpoint_profile.compute",
        title="Compute weighted monotone subsequence endpoint profiles",
        description=(
            "For one bounded ordered finite word with one nonnegative exact "
            "rational weight per position, return the two exact endpoint "
            "dynamic-programming profiles: S_i (weakly increasing) and T_i "
            "(weakly decreasing)."
        ),
        request_type=EndpointProfileRequest,
        result_type=EndpointProfileResult,
        run=compute_endpoint_profile_op,
        tags=("algebraic-combinatorics", "exact"),
        examples=(
            OperationExample(
                name="simple_word",
                description="Word 'ab' with weights 1,2.",
                input={
                    "source": {
                        "word": {
                            "alphabet": ["a", "b"],
                            "letters": ["a", "b"],
                        },
                        "weights": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_combinatorics.weighted_monotone_subsequence.nondecreasing_maximum.compute",
        title="Compute a maximum-weight weakly increasing subsequence",
        description=(
            "For a bounded finite word with one nonnegative exact rational "
            "weight per position, return the exact maximum total weight of a "
            "nondecreasing subsequence and one deterministic source-index "
            "witness. Equal adjacent letters are allowed; the exact input word, "
            "weight, positions, and values are retained."
        ),
        request_type=WeightedMaximumRequest,
        result_type=WeightedMaximumResult,
        run=maximum_weight_nondecreasing_subsequence_op,
        tags=("algebraic-combinatorics", "subsequences", "exact", "weighted"),
        discovery_terms=(
            "maximum weight nondecreasing subsequence",
            "weighted weakly increasing subsequence",
        ),
        examples=(
            OperationExample(
                name="weighted_nondecreasing",
                description="The full word ab has weight 3 and is nondecreasing.",
                input={
                    "source": {
                        "word": {"alphabet": ["a", "b"], "letters": ["a", "b"]},
                        "weights": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_combinatorics.weighted_monotone_subsequence.nonincreasing_maximum.compute",
        title="Compute a maximum-weight weakly decreasing subsequence",
        description=(
            "For a bounded finite word with one nonnegative exact rational "
            "weight per position, return the exact maximum total weight of a "
            "nonincreasing subsequence and one deterministic source-index "
            "witness. Equal adjacent letters are allowed; the exact input word, "
            "weight, positions, and values are retained."
        ),
        request_type=WeightedMaximumRequest,
        result_type=WeightedMaximumResult,
        run=maximum_weight_nonincreasing_subsequence_op,
        tags=("algebraic-combinatorics", "subsequences", "exact", "weighted"),
        discovery_terms=(
            "maximum weight nonincreasing subsequence",
            "weighted weakly decreasing subsequence",
        ),
        examples=(
            OperationExample(
                name="weighted_nonincreasing",
                description="In ba with weights 1,2, both letters contribute weight 3.",
                input={
                    "source": {
                        "word": {"alphabet": ["a", "b"], "letters": ["b", "a"]},
                        "weights": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
