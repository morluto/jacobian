"""Weighted monotone subsequence endpoint profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.algebraic.weighted_monotone._models import (
    EndpointProfileRequest,
    EndpointProfileResult,
)
from jacobian.math.combinatorics.algebraic.weighted_monotone.operations import (
    compute_endpoint_profile,
)


def compute_endpoint_profile_op(
    request: EndpointProfileRequest,
) -> EndpointProfileResult:
    return compute_endpoint_profile(request.source)


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
)

__all__ = ["TOOLS"]
