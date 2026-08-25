"""Chain complex operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.chain_complexes._models import (
    ComputeHomologyRequest,
    ConstructChainComplexRequest,
    MappingConeRequest,
    TensorProductRequest,
    VerifyChainMapRequest,
    VerifyDifferentialRequest,
)
from jacobian.math.chain_complexes.operations import (
    compute_homology,
    compute_mapping_cone,
    compute_tensor_product,
    construct_chain_complex,
    verify_chain_map,
    verify_differential,
)
from jacobian.math.chain_complexes.values import (
    ChainComplexValue,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
    VerificationResult,
)

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
        run=construct_chain_complex,
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
        run=verify_differential,
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
        run=verify_chain_map,
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
        run=compute_homology,
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
        run=compute_mapping_cone,
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
        run=compute_tensor_product,
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
