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
from jacobian.math.topology.frames.operations import (
    _gram_minimum_result_bound,
    _gram_result,
    _gram_result_bytes,
    _require_gram_entry_representation,
    _require_gram_work_budget,
    _require_result_budget,
    coherence,
    frame_potential,
)


def _gram(request: VectorFamilyRequest) -> GramResult:
    _require_gram_work_budget(request)
    _require_gram_entry_representation(request)
    _require_result_budget(_gram_minimum_result_bound(request))
    result = _gram_result(request)
    _require_result_budget(_gram_result_bytes(result))
    return result


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
        description="Compute the maximum normalized off-diagonal Gram entry "
        "after checking that the family spans the ambient space.",
        request_type=CoherenceRequest,
        result_type=CoherenceResult,
        run=_coherence,
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
        description="Compute the exact frame potential sum_{i,j} |<v_i, v_j>|^2 "
        "after checking that the family spans the ambient space.",
        request_type=FiniteFrameRequest,
        result_type=FramePotentialResult,
        run=_frame_potential,
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
