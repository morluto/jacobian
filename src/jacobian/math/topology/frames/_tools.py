"""Immutable declarations for finite-frame operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    ComplexFrameDesignProfileRequest,
    ComplexFrameDesignProfileResult,
    FramePotentialResult,
    GramResult,
    MutuallyUnbiasedBasesRequest,
    MutuallyUnbiasedBasesResult,
    SicProfileRequest,
    SicProfileResult,
    TightEquiangularProfileResult,
)
from jacobian.math.topology.frames.operations import (
    coherence,
    complex_design_profile,
    frame_potential,
    gram,
    mutually_unbiased_bases,
    sic_profile,
    tight_equiangular_profile,
)
from jacobian.math.topology.frames.values import VectorFamily


def _gram(request: VectorFamily) -> GramResult:
    return gram(request)


def _coherence(request: VectorFamily) -> CoherenceResult:
    return coherence(request)


def _frame_potential(request: VectorFamily) -> FramePotentialResult:
    return frame_potential(request)


def _tight_equiangular_profile(request: VectorFamily) -> TightEquiangularProfileResult:
    return tight_equiangular_profile(request)


def _complex_design_profile(
    request: ComplexFrameDesignProfileRequest,
) -> ComplexFrameDesignProfileResult:
    return complex_design_profile(request)


def _mutually_unbiased_bases(
    request: MutuallyUnbiasedBasesRequest,
) -> MutuallyUnbiasedBasesResult:
    return mutually_unbiased_bases(request)


def _sic_profile(request: SicProfileRequest) -> SicProfileResult:
    return sic_profile(request)


_ORTHONORMAL = {"dimension": 2, "vectors": [[1, 0], [0, 1]]}
_C0 = {"real": {"num": "0", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_C1 = {"real": {"num": "1", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_CM1 = {"real": {"num": "-1", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_COMPLEX_STANDARD = {
    "dimension": 2,
    "vectors": [[_C1, _C0], [_C0, _C1]],
}
_COMPLEX_HADAMARD = {
    "dimension": 2,
    "vectors": [[_C1, _C1], [_C1, _CM1]],
}

TOOLS: MathTools = (
    MathTool(
        operation_id="frame.complex_design_profile.compute",
        title="Classify an exact complex frame design",
        description="Return exact tightness and equal-norm equiangularity for a Gaussian-rational complex frame.",
        request_type=ComplexFrameDesignProfileRequest,
        result_type=ComplexFrameDesignProfileResult,
        run=_complex_design_profile,
        tags=("topology", "frame", "design", "complex", "exact"),
        examples=(OperationExample(name="complex_standard", description="Profile the standard complex basis.", input={"frame": _COMPLEX_STANDARD}),),
    ),
    MathTool(
        operation_id="frame.mutually_unbiased_bases.compute",
        title="Decide mutual unbiasedness of exact complex bases",
        description="Check orthogonality within each basis and the exact normalized cross-overlap equation between every basis pair.",
        request_type=MutuallyUnbiasedBasesRequest,
        result_type=MutuallyUnbiasedBasesResult,
        run=_mutually_unbiased_bases,
        tags=("topology", "frame", "MUB", "complex", "exact"),
        examples=(OperationExample(name="standard_and_hadamard", description="Two mutually unbiased real bases in dimension two.", input={"dimension": 2, "bases": [_COMPLEX_STANDARD, _COMPLEX_HADAMARD]}),),
    ),
    MathTool(
        operation_id="frame.sic_profile.compute",
        title="Decide the exact SIC overlap equations",
        description="Check the d-squared cardinality, equal norms, and 1/(d+1) pairwise normalized overlap equation for an exact complex frame.",
        request_type=SicProfileRequest,
        result_type=SicProfileResult,
        run=_sic_profile,
        tags=("topology", "frame", "SIC", "complex", "exact"),
        examples=(OperationExample(name="dimension_one_sic", description="The canonical one-dimensional SIC profile.", input={"frame": {"dimension": 1, "vectors": [[_C1]]}}),),
    ),
    MathTool(
        operation_id="frame.tight_equiangular_profile.compute",
        title="Classify tightness and equiangularity of a frame",
        description="Return exact tight-frame and equiangular-frame predicates for an integer vector family.",
        request_type=VectorFamily,
        result_type=TightEquiangularProfileResult,
        run=_tight_equiangular_profile,
        tags=("topology", "frame", "tight", "equiangular", "exact"),
        examples=(OperationExample(name="orthonormal_frame", description="Classify an orthonormal frame.", input=_ORTHONORMAL),),
    ),
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
