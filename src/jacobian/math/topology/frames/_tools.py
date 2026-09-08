"""Immutable declarations for finite-frame operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    GramResult,
)
from jacobian.math.topology.frames.operations import (
    coherence,
    frame_potential,
    gram,
)
from jacobian.math.topology.frames.values import VectorFamily


def _gram(request: VectorFamily) -> GramResult:
    return gram(request)


def _coherence(request: VectorFamily) -> CoherenceResult:
    return coherence(request)


def _frame_potential(request: VectorFamily) -> FramePotentialResult:
    return frame_potential(request)


_ORTHONORMAL = {"dimension": 2, "vectors": [[1, 0], [0, 1]]}

TOOLS: MathTools = (
    MathTool(
        operation_id="frame.gram.compute",
        title="Compute the Gram matrix of a vector family",
        description="Compute the exact Gram matrix G with G_ij = <v_i, v_j> "
        "for a finite family of integer vectors.",
        request_type=VectorFamily,
        result_type=GramResult,
        run=_gram,
        tags=("topology", "frame", "gram", "exact"),
        examples=(
            OperationExample(
                name="orthonormal_frame",
                description="Gram matrix of an orthonormal frame.",
                input=_ORTHONORMAL,
            ),
        ),
    ),
    MathTool(
        operation_id="frame.coherence.compute",
        title="Compute the coherence of a frame",
        description="Compute the maximum normalized off-diagonal Gram entry "
        "after checking that the family spans the ambient space.",
        request_type=VectorFamily,
        result_type=CoherenceResult,
        run=_coherence,
        tags=("topology", "frame", "coherence", "exact"),
        examples=(
            OperationExample(
                name="orthonormal_frame",
                description="Coherence of an orthonormal frame.",
                input=_ORTHONORMAL,
            ),
        ),
    ),
    MathTool(
        operation_id="frame.potential.compute",
        title="Compute the frame potential",
        description="Compute the exact frame potential sum_{i,j} |<v_i, v_j>|^2 "
        "after checking that the family spans the ambient space.",
        request_type=VectorFamily,
        result_type=FramePotentialResult,
        run=_frame_potential,
        tags=("topology", "frame", "potential", "exact"),
        examples=(
            OperationExample(
                name="orthonormal_frame",
                description="Frame potential of an orthonormal frame.",
                input=_ORTHONORMAL,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
