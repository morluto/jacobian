"""Immutable declarations for finite-frame operations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    ComplexFrameProfileRequest,
    ComplexFrameProfileResult,
    CyclotomicSicPovmRequest,
    CyclotomicSicPovmResult,
    CyclotomicSicRequest,
    CyclotomicSicResult,
    FramePotentialResult,
    GramResult,
    MutuallyUnbiasedBasesRequest,
    MutuallyUnbiasedBasesResult,
    ProjectiveDesignRequest,
    ProjectiveDesignResult,
    SicProfileRequest,
    SicProfileResult,
    SphericalDesignRequest,
    SphericalDesignResult,
    TightEquiangularProfileResult,
)
from jacobian.math.topology.frames.operations import (
    coherence,
    complex_frame_profile,
    cyclotomic_sic_povm,
    cyclotomic_sic_profile,
    frame_potential,
    gram,
    mutually_unbiased_bases,
    projective_design_profile,
    sic_profile,
    spherical_design_verify,
    tight_equiangular_profile,
)
from jacobian.math.topology.frames.values import VectorFamily


def _gram(request: VectorFamily) -> GramResult:
    return gram(request)


def _coherence(request: VectorFamily) -> CoherenceResult:
    return coherence(request)


def _frame_potential(request: VectorFamily) -> FramePotentialResult:
    return frame_potential(request)


def _require_request(request: object, expected: type[object]) -> None:
    if type(request) is not expected:
        raise OperationDomainValidationError(
            location=(),
            code="frames.request_type",
            message=f"operation requires a {expected.__name__} request",
        )


def _tight_equiangular_profile(request: VectorFamily) -> TightEquiangularProfileResult:
    return tight_equiangular_profile(request)


def _complex_frame_profile(
    request: ComplexFrameProfileRequest,
) -> ComplexFrameProfileResult:
    _require_request(request, ComplexFrameProfileRequest)
    return complex_frame_profile(request.frame)


def _mutually_unbiased_bases(
    request: MutuallyUnbiasedBasesRequest,
) -> MutuallyUnbiasedBasesResult:
    _require_request(request, MutuallyUnbiasedBasesRequest)
    return mutually_unbiased_bases(request.dimension, request.bases)


def _sic_profile(request: SicProfileRequest) -> SicProfileResult:
    _require_request(request, SicProfileRequest)
    return sic_profile(request.frame)


def _spherical_design(request: SphericalDesignRequest) -> SphericalDesignResult:
    return spherical_design_verify(request.family, request.weights, request.strength)


def _projective_design(request: ProjectiveDesignRequest) -> ProjectiveDesignResult:
    return projective_design_profile(request.frame, request.weights, request.strength)


def _cyclotomic_sic(request: CyclotomicSicRequest) -> CyclotomicSicResult:
    return cyclotomic_sic_profile(request.frame)


def _cyclotomic_sic_povm(request: CyclotomicSicPovmRequest) -> CyclotomicSicPovmResult:
    return cyclotomic_sic_povm(request.frame)


_ORTHONORMAL = {"dimension": 2, "vectors": [[1, 0], [0, 1]]}
_OCTAHEDRON = {
    "dimension": 3,
    "vectors": [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
}
_OCTAHEDRON_WEIGHTS = [{"num": "1", "den": "6"}] * 6
_C0 = {"real": {"num": "0", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_C1 = {"real": {"num": "1", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_CM1 = {"real": {"num": "-1", "den": "1"}, "imaginary": {"num": "0", "den": "1"}}
_UNIT_SCALAR = {"order": 1, "coefficients": [{"num": "1", "den": "1"}]}
_UNIT_SIC = {"order": 1, "dimension": 1, "vectors": [[_UNIT_SCALAR]]}
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
        operation_id="frame.complex_profile.compute",
        title="Profile an exact complex frame",
        description="Return the exact frame operator, tight residual, and equal-norm equiangular profile for a Gaussian-rational complex frame.",
        request_type=ComplexFrameProfileRequest,
        result_type=ComplexFrameProfileResult,
        run=_complex_frame_profile,
        tags=("topology", "frame", "complex", "exact"),
        examples=(
            OperationExample(
                name="complex_standard",
                description="Profile the standard complex basis.",
                input={"frame": _COMPLEX_STANDARD},
            ),
        ),
    ),
    MathTool(
        operation_id="frame.mutually_unbiased_bases.compute",
        title="Decide mutual unbiasedness of exact complex bases",
        description="Check nonzero orthogonality within each basis and the exact normalized cross-overlap equation between every basis pair; unit coordinate norms are not required.",
        request_type=MutuallyUnbiasedBasesRequest,
        result_type=MutuallyUnbiasedBasesResult,
        run=_mutually_unbiased_bases,
        tags=("topology", "frame", "MUB", "complex", "exact"),
        examples=(
            OperationExample(
                name="standard_and_hadamard",
                description="Two mutually unbiased real bases in dimension two.",
                input={"dimension": 2, "bases": [_COMPLEX_STANDARD, _COMPLEX_HADAMARD]},
            ),
        ),
    ),
    MathTool(
        operation_id="frame.sic_profile.compute",
        title="Decide the exact SIC overlap equations",
        description="Check the d-squared cardinality and 1/(d+1) pairwise normalized overlap equation for nonzero complex projective representatives, using normalized rank-one projectors.",
        request_type=SicProfileRequest,
        result_type=SicProfileResult,
        run=_sic_profile,
        tags=("topology", "frame", "SIC", "complex", "exact"),
        examples=(
            OperationExample(
                name="dimension_one_sic",
                description="The canonical one-dimensional SIC profile.",
                input={"frame": {"dimension": 1, "vectors": [[_C1]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="frame.tight_equiangular_profile.compute",
        title="Classify tightness and equiangularity of a frame",
        description="Return exact tight-frame and equiangular-frame predicates for an integer vector family.",
        request_type=VectorFamily,
        result_type=TightEquiangularProfileResult,
        run=_tight_equiangular_profile,
        tags=("topology", "frame", "tight", "equiangular", "exact"),
        examples=(
            OperationExample(
                name="orthonormal_frame",
                description="Classify an orthonormal frame.",
                input=_ORTHONORMAL,
            ),
        ),
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
    MathTool(
        operation_id="frame.spherical_design.verify",
        title="Verify a weighted spherical design",
        description="Check weighted cubature exactness against exact sphere "
        "moments through the declared strength; weights must be nonnegative "
        "and sum to one, and vector phase is irrelevant to the verdict.",
        request_type=SphericalDesignRequest,
        result_type=SphericalDesignResult,
        run=_spherical_design,
        tags=("topology", "frame", "spherical-design", "exact"),
        examples=(
            OperationExample(
                name="octahedron_design_3",
                description="The octahedron with uniform weights is a spherical 3-design; weights must sum to one.",
                input={
                    "family": _OCTAHEDRON,
                    "weights": _OCTAHEDRON_WEIGHTS,
                    "strength": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="frame.projective_design.profile",
        title="Decide a weighted projective design identity",
        description="Check the weighted Welch identity for normalized complex "
        "projective representatives through the declared strength; overlaps "
        "are scale-invariant so vector phases never affect the verdict.",
        request_type=ProjectiveDesignRequest,
        result_type=ProjectiveDesignResult,
        run=_projective_design,
        tags=("topology", "frame", "projective-design", "exact"),
        examples=(
            OperationExample(
                name="qubit_basis_design_1",
                description="The qubit computational basis is a projective 1-design; weights must sum to one.",
                input={
                    "frame": _COMPLEX_STANDARD,
                    "weights": [
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "2"},
                    ],
                    "strength": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="frame.cyclotomic_sic.profile",
        title="Decide SIC equations over cyclotomic representatives",
        description="Check d-squared cardinality and exact 1/(d+1) normalized "
        "overlaps over a declared cyclotomic order; dimension two needs "
        "sqrt(3) and is carried exactly at order 24.",
        request_type=CyclotomicSicRequest,
        result_type=CyclotomicSicResult,
        run=_cyclotomic_sic,
        tags=("topology", "frame", "SIC", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="dimension_one_cyclotomic_sic",
                description="The one-line cyclotomic SIC; the frame must use one cyclotomic order.",
                input={"frame": _UNIT_SIC},
            ),
        ),
    ),
    MathTool(
        operation_id="frame.cyclotomic_sic.povm.compute",
        title="Compute exact SIC POVM effects",
        description="Return SIC effects over one common cyclotomic denominator "
        "with projector identities and identity resolution verified in the "
        "kernel; retained matrices compose unchanged with a quantum consumer.",
        request_type=CyclotomicSicPovmRequest,
        result_type=CyclotomicSicPovmResult,
        run=_cyclotomic_sic_povm,
        tags=("topology", "frame", "SIC", "POVM", "exact"),
        examples=(
            OperationExample(
                name="dimension_one_povm",
                description="The one-line SIC POVM is the scalar 1; the source must be an exact SIC.",
                input={"frame": _UNIT_SIC},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
