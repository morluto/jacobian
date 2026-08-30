"""Weighted monotone subsequence endpoint profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def wms_action[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    wms_action(
        "algebraic_combinatorics.weighted_monotone_subsequence.endpoint_profile.compute",
        "Compute weighted monotone subsequence endpoint profiles",
        (
            "For one bounded ordered finite word with one nonnegative exact "
            "rational weight per position, return the two exact endpoint "
            "dynamic-programming profiles: S_i (weakly increasing) and T_i "
            "(weakly decreasing)."
        ),
        EndpointProfileRequest,
        EndpointProfileResult,
        compute_endpoint_profile_op,
        "algebraic-combinatorics",
        "exact",
        examples=(
            example(
                "simple_word",
                "Word 'ab' with weights 1,2.",
                {
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
