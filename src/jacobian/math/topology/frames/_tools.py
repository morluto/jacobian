"""Immutable declarations for finite-frame operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    CoherenceResult,
    FiniteFrameRequest,
    FramePotentialResult,
    GramResult,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames.operations import (
    coherence,
    frame_potential,
    gram,
)


def _gram(request: VectorFamilyRequest) -> GramResult:
    return gram(request)


def _coherence(request: CoherenceRequest) -> CoherenceResult:
    return coherence(request)


def _frame_potential(request: FiniteFrameRequest) -> FramePotentialResult:
    return frame_potential(request)


_ORTHONORMAL = {"vectors": [[1, 0], [0, 1]]}

TOOLS: MathTools = (
    MathTool(
        operation_id="frame.gram.compute",
        title="Compute the Gram matrix of a vector family",
        description="Compute the exact Gram matrix G with G_ij = <v_i, v_j> "
        "for a finite family of integer vectors.",
        request_type=VectorFamilyRequest,
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
        request_type=CoherenceRequest,
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
        request_type=FiniteFrameRequest,
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
