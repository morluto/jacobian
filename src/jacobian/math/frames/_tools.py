"""Frame operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.frames._models import (
    CoherenceRequest,
    CoherenceResult,
    FiniteFrameRequest,
    FramePotentialResult,
    GramResult,
    VectorFamilyRequest,
)
from jacobian.math.frames._operations import (
    compute_coherence,
    compute_frame_potential,
    compute_gram,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "frame.gram.compute",
        "Compute the Gram matrix of a frame",
        "Compute the Gram matrix G with G_ij = <v_i, v_j> for a finite frame.",
        VectorFamilyRequest,
        GramResult,
        compute_gram,
        "frame",
        "gram",
        "exact",
        examples=(
            example(
                "orthonormal_frame",
                "Gram matrix of an orthonormal frame.",
                {"vectors": [[1, 0], [0, 1]]},
            ),
        ),
    ),
    _op(
        "frame.coherence.compute",
        "Compute the coherence of a frame",
        "Compute the frame coherence as the maximum normalized off-diagonal Gram entry.",
        CoherenceRequest,
        CoherenceResult,
        compute_coherence,
        "frame",
        "coherence",
        "exact",
        examples=(
            example(
                "orthonormal_frame",
                "Coherence of an orthonormal frame.",
                {"vectors": [[1, 0], [0, 1]]},
            ),
        ),
    ),
    _op(
        "frame.potential.compute",
        "Compute the frame potential",
        "Compute the frame potential sum_{i,j} |<v_i, v_j>|^2.",
        FiniteFrameRequest,
        FramePotentialResult,
        compute_frame_potential,
        "frame",
        "potential",
        "exact",
        examples=(
            example(
                "orthonormal_frame",
                "Frame potential of an orthonormal frame.",
                {"vectors": [[1, 0], [0, 1]]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
