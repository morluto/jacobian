"""Immutable declarations for finite-frame operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    CoherenceResult,
    FiniteFrameRequest,
    FramePotentialResult,
    GramResult,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames._operations import (
    compute_coherence,
    compute_frame_potential,
    compute_gram,
)

_ORTHONORMAL = {"vectors": [[1, 0], [0, 1]]}

TOOLS: MathTools = (
    MathTool(
        operation_id="frame.gram.compute",
        title="Compute the Gram matrix of a vector family",
        description="Compute the Gram matrix G with G_ij = <v_i, v_j>.",
        request_type=VectorFamilyRequest,
        result_type=GramResult,
        run=compute_gram,
        tags=("topology", "frame", "gram", "exact"),
        examples=(
            example(
                "orthonormal_frame",
                "Gram matrix of an orthonormal frame.",
                _ORTHONORMAL,
            ),
        ),
    ),
    MathTool(
        operation_id="frame.coherence.compute",
        title="Compute the coherence of a frame",
        description="Compute the maximum normalized off-diagonal Gram entry.",
        request_type=CoherenceRequest,
        result_type=CoherenceResult,
        run=compute_coherence,
        tags=("topology", "frame", "coherence", "exact"),
        examples=(
            example(
                "orthonormal_frame",
                "Coherence of an orthonormal frame.",
                _ORTHONORMAL,
            ),
        ),
    ),
    MathTool(
        operation_id="frame.potential.compute",
        title="Compute the frame potential",
        description="Compute the frame potential sum_{i,j} |<v_i, v_j>|^2.",
        request_type=FiniteFrameRequest,
        result_type=FramePotentialResult,
        run=compute_frame_potential,
        tags=("topology", "frame", "potential", "exact"),
        examples=(
            example(
                "orthonormal_frame",
                "Frame potential of an orthonormal frame.",
                _ORTHONORMAL,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
