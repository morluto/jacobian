"""Chain complex operation declarations."""

from pydantic import ValidationError

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.topology.chain_complexes._filtered_models import (
    AssociatedGradedResult,
    FilteredChainComplexRequest,
    SpectralPageRequest,
    SpectralPageResult,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    associated_graded as _associated_graded_native,
)
from jacobian.math.topology.chain_complexes._filtered_operations import (
    spectral_page as _spectral_page_native,
)
from jacobian.math.topology.chain_complexes._models import (
    ComputeHomologyRequest,
    ConstructChainComplexRequest,
    MappingConeRequest,
    TensorProductRequest,
    VerifyChainMapRequest,
    VerifyDifferentialRequest,
)
from jacobian.math.topology.chain_complexes.operations import (
    chain_map_commutes,
    construct_chain_complex,
    differential_squares_to_zero,
    homology_groups,
    mapping_cone,
    tensor_product_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
    VerificationResult,
)


def _construct(request: ConstructChainComplexRequest) -> ChainComplexValue:
    """Project a wire request into the canonical construction operation."""
    try:
        return construct_chain_complex(
            request.basis_sizes,
            request.differential_matrices,
            coefficient_ring=request.coefficient_ring,
            prime=request.prime,
        )
    except ValidationError as exc:
        error = exc.errors(include_url=False, include_context=False)[0]
        location = tuple(error.get("loc", ())) or ("differential_matrices",)
        raise OperationDomainValidationError(
            location=location,
            code=str(error["type"]),
            message=str(error["msg"]),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("differential_matrices",),
            code="chain_complex.differential_not_square_zero",
            message=str(exc),
        ) from exc


def _verify_differential(request: VerifyDifferentialRequest) -> VerificationResult:
    """Project a wire request into the canonical differential verifier."""
    return differential_squares_to_zero(request.complex)


def _verify_chain_map(request: VerifyChainMapRequest) -> VerificationResult:
    """Project a wire request into the canonical chain-map verifier."""
    return chain_map_commutes(request.source, request.target, request.map_matrices)


def _homology(request: ComputeHomologyRequest) -> HomologyResult:
    """Project a wire request into the canonical homology operation."""
    return homology_groups(request.complex)


def _mapping_cone(request: MappingConeRequest) -> MappingConeResult:
    """Project a wire request into the canonical mapping-cone operation."""
    try:
        return mapping_cone(request.source, request.target, request.map_matrices)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("map_matrices",),
            code="chain_complex.chain_map_relation",
            message=str(exc),
        ) from exc


def _tensor_product(request: TensorProductRequest) -> TensorProductResult:
    """Project a wire request into the canonical tensor operation."""
    return tensor_product_complex(request.left, request.right)


def _associated_graded(request: FilteredChainComplexRequest) -> AssociatedGradedResult:
    """Project a wire request into the canonical associated-graded operation."""
    return _associated_graded_native(request.complex, request.filtration)


def _spectral_page(request: SpectralPageRequest) -> SpectralPageResult:
    """Project a wire request into the canonical spectral-page operation."""
    return _spectral_page_native(request.complex, request.filtration, request.page)


_CIRCLE_COMPLEX = {
    "coefficient_ring": "QQ",
    "degree_min": 0,
    "degree_max": 1,
    "basis_sizes": [3, 3],
    "differential_matrices": [
        [["-1", "1", "0"], ["0", "-1", "1"], ["0", "0", "0"]],
    ],
}

_MULTIPLICATION_BY_SIX_COMPLEX = {
    "coefficient_ring": "ZZ",
    "degree_min": 0,
    "degree_max": 1,
    "basis_sizes": [1, 1],
    "differential_matrices": [[["6"]]],
}


TOOLS: MathTools = (
    MathTool(
        operation_id="chain_complex.construct.compute",
        title="Construct a finite based chain complex",
        description=(
            "Construct a bounded homological chain complex from differential "
            "matrices over ZZ, QQ, or a prime field."
        ),
        request_type=ConstructChainComplexRequest,
        result_type=ChainComplexValue,
        run=_construct,
        tags=("chain-complex", "exact"),
        examples=(
            OperationExample(
                name="circle_chain_complex",
                description="Construct the chain complex of a circle (3 edges, 3 "
                "vertices). Supply exactly one fewer differential matrix "
                "than basis sizes; matrix i must have shape basis_sizes[i] "
                "x basis_sizes[i+1], and adjacent matrices must compose to "
                "zero (d^2 = 0).",
                input={
                    "coefficient_ring": "QQ",
                    "basis_sizes": [3, 3],
                    "differential_matrices": [
                        [["-1", "1", "0"], ["0", "-1", "1"], ["0", "0", "0"]],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="chain_complex.verify_differential.compute",
        title="Verify d^2 = 0",
        description="Verify that the differential of a chain complex squares to zero.",
        request_type=VerifyDifferentialRequest,
        result_type=VerificationResult,
        run=_verify_differential,
        tags=("chain-complex", "exact"),
        examples=(
            OperationExample(
                name="verify_circle_d2",
                description="Verify d^2 = 0 for the circle chain complex.",
                input={"complex": _CIRCLE_COMPLEX},
            ),
        ),
    ),
    MathTool(
        operation_id="chain_complex.verify_chain_map.compute",
        title="Verify a chain map commutes",
        description="Verify that a chain map f: C -> D commutes with differentials.",
        request_type=VerifyChainMapRequest,
        result_type=VerificationResult,
        run=_verify_chain_map,
        tags=("chain-complex", "exact"),
        examples=(
            OperationExample(
                name="verify_identity_map",
                description="Verify the identity map commutes.",
                input={
                    "source": _CIRCLE_COMPLEX,
                    "target": _CIRCLE_COMPLEX,
                    "map_matrices": [
                        [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                        [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="chain_complex.homology.compute",
        title="Compute homology of a chain complex",
        description=(
            "Compute exact homology of a bounded finite based chain complex. "
            "Over QQ or GF(p), return vector-space ranks. Over ZZ, return "
            "finitely generated abelian groups with invariant factors, "
            "source-basis free and torsion cycles, torsion bounding chains, "
            "and both Smith transformation certificates in every degree."
        ),
        request_type=ComputeHomologyRequest,
        result_type=HomologyResult,
        run=_homology,
        tags=(
            "chain-complex",
            "homology",
            "integer-homology",
            "torsion",
            "smith-normal-form",
            "exact",
        ),
        examples=(
            OperationExample(
                name="circle_homology",
                description="Compute homology of the circle (Betti numbers 1, 1).",
                input={"complex": _CIRCLE_COMPLEX},
            ),
            OperationExample(
                name="multiplication_by_six_homology",
                description="Compute H_0 = Z/6 and its torsion cycle and bounding chain.",
                input={"complex": _MULTIPLICATION_BY_SIX_COMPLEX},
            ),
        ),
    ),
    MathTool(
        operation_id="chain_complex.mapping_cone.compute",
        title="Compute the mapping cone",
        description="Compute the mapping cone of a chain map f: C -> D.",
        request_type=MappingConeRequest,
        result_type=MappingConeResult,
        run=_mapping_cone,
        tags=("chain-complex", "exact"),
        examples=(
            OperationExample(
                name="identity_mapping_cone",
                description="Mapping cone of the identity on a circle.",
                input={
                    "source": _CIRCLE_COMPLEX,
                    "target": _CIRCLE_COMPLEX,
                    "map_matrices": [
                        [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                        [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="chain_complex.tensor_product.compute",
        title="Compute the tensor product of two chain complexes",
        description="Compute the tensor product (C ⊗ D)_n = ⊕_{i+j=n} C_i ⊗ D_j.",
        request_type=TensorProductRequest,
        result_type=TensorProductResult,
        run=_tensor_product,
        tags=("chain-complex", "exact"),
        examples=(
            OperationExample(
                name="tensor_two_circles",
                description="Tensor product of two circle chain complexes.",
                input={"left": _CIRCLE_COMPLEX, "right": _CIRCLE_COMPLEX},
            ),
        ),
    ),
    MathTool(
        operation_id="homological.spectral_sequence.page.compute",
        title="Compute one page of the spectral sequence of a filtered complex",
        description=(
            "Compute the E^r page of a finite bounded increasing QQ or GF(p) "
            "filtered chain complex as bigraded modules with explicit "
            "quotient representatives and the induced d^r differentials of "
            "bidegree (-r, r - 1); d^r d^r = 0 is replayed inside the "
            "kernel. Page 0 is the associated graded. The result previews "
            "the next-page dimensions as the homology of (E^r, d^r) and "
            "reports STABILIZED only when every longer differential "
            "provably vanishes, TRUNCATED when the page budget ceiling cuts "
            "the sequence first, and ACTIVE otherwise."
        ),
        request_type=SpectralPageRequest,
        result_type=SpectralPageResult,
        run=_spectral_page,
        tags=("chain-complex", "filtered-complex", "spectral-sequence", "exact"),
        discovery_terms=(
            "spectral sequence page",
            "spectral sequence differentials",
            "filtered complex E1 E2",
            "spectral sequence collapse",
        ),
        examples=(
            OperationExample(
                name="two_step_spectral_page_one",
                description=(
                    "Compute the E^1 page of a two-term complex with a zero "
                    "bottom level and an exhaustive top level; the d^1 map "
                    "recovers the connecting differential and the next-page "
                    "preview shows the collapse."
                ),
                input={
                    "complex": {
                        "coefficient_ring": "QQ",
                        "degree_min": 0,
                        "degree_max": 1,
                        "basis_sizes": [1, 1],
                        "differential_matrices": [[["1"]]],
                    },
                    "filtration": [
                        {
                            "subspaces": [
                                {"vectors": [["1"]]},
                                {"vectors": []},
                            ]
                        },
                        {
                            "subspaces": [
                                {"vectors": [["1"]]},
                                {"vectors": [["1"]]},
                            ]
                        },
                    ],
                    "page": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="homological.filtered_chain_complex.associated_graded.compute",
        title="Compute the associated graded of a filtered chain complex",
        description=(
            "Compute Gr_p C_n = F_p C_n / F_{p-1} C_n with the induced "
            "degree-preserving differentials for a finite bounded increasing "
            "QQ or GF(p) filtration; the top level must span every chain "
            "group and the differential must preserve each level."
        ),
        request_type=FilteredChainComplexRequest,
        result_type=AssociatedGradedResult,
        run=_associated_graded,
        tags=("chain-complex", "filtered-complex", "associated-graded", "exact"),
        discovery_terms=(
            "filtered chain complex",
            "associated graded",
            "filtration quotient",
            "graded differential",
        ),
        examples=(
            OperationExample(
                name="two_step_associated_graded",
                description=(
                    "Compute the associated graded of a two-term complex with "
                    "a zero bottom level and an exhaustive top level; the "
                    "top graded piece recovers the source differential."
                ),
                input={
                    "complex": {
                        "coefficient_ring": "QQ",
                        "degree_min": 0,
                        "degree_max": 1,
                        "basis_sizes": [1, 1],
                        "differential_matrices": [[["1"]]],
                    },
                    "filtration": [
                        {"subspaces": [{"vectors": []}, {"vectors": []}]},
                        {
                            "subspaces": [
                                {"vectors": [["1"]]},
                                {"vectors": [["1"]]},
                            ]
                        },
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
