"""Chain complex operation declarations."""

from pydantic import ValidationError

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationDomainValidationError
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
            coefficient_field=request.coefficient_field,
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
    return mapping_cone(request.source, request.target, request.map_matrices)


def _tensor_product(request: TensorProductRequest) -> TensorProductResult:
    """Project a wire request into the canonical tensor operation."""
    return tensor_product_complex(request.left, request.right)


_CIRCLE_COMPLEX = {
    "coefficient_field": "QQ",
    "degree_min": 0,
    "degree_max": 1,
    "basis_sizes": [3, 3],
    "differential_matrices": [
        [["-1", "1", "0"], ["0", "-1", "1"], ["0", "0", "0"]],
    ],
}


TOOLS: MathTools = (
    MathTool(
        operation_id="chain_complex.construct.compute",
        title="Construct a finite based chain complex",
        description=(
            "Construct a bounded homological chain complex from differential "
            "matrices over QQ or a prime field."
        ),
        request_type=ConstructChainComplexRequest,
        result_type=ChainComplexValue,
        run=_construct,
        tags=("chain-complex", "exact"),
        examples=(
            example(
                "circle_chain_complex",
                "Construct the chain complex of a circle (3 edges, 3 "
                "vertices). Supply exactly one fewer differential matrix "
                "than basis sizes; matrix i must have shape basis_sizes[i] "
                "x basis_sizes[i+1], and adjacent matrices must compose to "
                "zero (d^2 = 0).",
                {
                    "coefficient_field": "QQ",
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
            example(
                "verify_circle_d2",
                "Verify d^2 = 0 for the circle chain complex.",
                {"complex": _CIRCLE_COMPLEX},
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
            example(
                "verify_identity_map",
                "Verify the identity map commutes.",
                {
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
            "Compute the homology groups (Betti numbers) of a bounded chain "
            "complex over QQ or a prime field using exact linear algebra."
        ),
        request_type=ComputeHomologyRequest,
        result_type=HomologyResult,
        run=_homology,
        tags=("chain-complex", "homology", "exact"),
        examples=(
            example(
                "circle_homology",
                "Compute homology of the circle (Betti numbers 1, 1).",
                {"complex": _CIRCLE_COMPLEX},
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
            example(
                "identity_mapping_cone",
                "Mapping cone of the identity on a circle.",
                {
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
            example(
                "tensor_two_circles",
                "Tensor product of two circle chain complexes.",
                {"left": _CIRCLE_COMPLEX, "right": _CIRCLE_COMPLEX},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
