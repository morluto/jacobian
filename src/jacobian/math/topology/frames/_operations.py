"""Wire adapters for finite vector-family and frame operations."""

from jacobian.math.topology.frames._models import (
    CoherenceRequest,
    CoherenceResult,
    FiniteFrameRequest,
    FramePotentialResult,
    GramResult,
    VectorFamilyRequest,
)
from jacobian.math.topology.frames.operations import coherence, frame_potential, gram


def compute_gram(request: VectorFamilyRequest) -> GramResult:
    return gram(request)


def compute_coherence(request: CoherenceRequest) -> CoherenceResult:
    return coherence(request)


def compute_frame_potential(request: FiniteFrameRequest) -> FramePotentialResult:
    return frame_potential(request)
