"""Typed declarations for cohomology operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cohomology.operations._models import (
    BocksteinRequest,
    BocksteinResult,
    CohomologyRingRequest,
    CohomologyRingResult,
    CupProductRequest,
    CupProductResult,
    InducedCohomologyMapRequest,
    InducedCohomologyMapResult,
    SteenrodSquareRequest,
    SteenrodSquareResult,
)
from jacobian.math.topology.cohomology.operations._simplicial import (
    SimplicialCohomologyRequest,
    SimplicialCohomologyResult,
    simplicial_cohomology,
)
from jacobian.math.topology.cohomology.operations.operations import (
    bockstein,
    cohomology_ring,
    cup_product,
    induced_cohomology_map,
    steenrod_square,
)


def _run_steenrod_square(request: SteenrodSquareRequest) -> SteenrodSquareResult:
    return steenrod_square(
        request.cochain_degree,
        request.simplex_values,
        request.simplex_coefficients,
        request.square_degree,
        request.ambient_simplices,
        request.ambient_complex,
    )


def _run_bockstein(request: BocksteinRequest) -> BocksteinResult:
    return bockstein(
        request.prime,
        request.cochain_degree,
        request.simplex_values,
        request.simplex_coefficients,
        request.ambient_simplices,
        request.ambient_complex,
    )


def _run_simplicial_cohomology(
    request: SimplicialCohomologyRequest,
) -> SimplicialCohomologyResult:
    return simplicial_cohomology(request.complex, request.prime, request.convention)


def _run_cup_product(request: CupProductRequest) -> CupProductResult:
    return cup_product(request.complex, request.prime, request.left, request.right)


def _run_cohomology_ring(request: CohomologyRingRequest) -> CohomologyRingResult:
    return cohomology_ring(request.complex, request.prime, request.convention)


def _run_induced_cohomology_map(
    request: InducedCohomologyMapRequest,
) -> InducedCohomologyMapResult:
    return induced_cohomology_map(request.map, request.prime, request.convention)


_SQ_EXAMPLE: dict[str, Any] = {
    "cochain_degree": 1,
    "simplex_values": [[0, 1], [0, 2]],
    "simplex_coefficients": [1, 1],
    "square_degree": 0,
    "ambient_simplices": [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]],
}

_BOCKSTEIN_EXAMPLE: dict[str, Any] = {
    "prime": 2,
    "cochain_degree": 1,
    "simplex_values": [[0, 1], [1, 2], [0, 2]],
    "simplex_coefficients": [2, 2, 2],
}


_CIRCLE_COMPLEX: dict[str, Any] = {
    "vertices": ["a", "b", "c"],
    "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
        {"dimension": 1, "faces": [["a", "b"], ["a", "c"], ["b", "c"]]},
    ],
    "dimension": 1,
    "f_vector": [3, 3],
    "closure_size": 6,
    "orientation_convention": "LEXICOGRAPHIC_VERTEX_ORDER",
    "empty_simplex_stored": False,
}


def _circle_cochain(coefficients: list[int]) -> dict[str, Any]:
    return {
        "complex": _CIRCLE_COMPLEX,
        "prime": 2,
        "degree": 1,
        "coefficients": coefficients,
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.simplicial.cohomology.compute",
        title="Compute prime-field simplicial cohomology with exact bases",
        description=(
            "Build the cochain groups C^k=Hom(C_k,GF(p)) with coboundary "
            "delta^k=transpose(boundary_{k+1}) for one bounded canonical "
            "finite simplicial complex and return per-degree cochain, "
            "cocycle, coboundary, and cohomology-class bases with "
            "dim H^k cross-checked against prime-field homology. The "
            "complex must carry its complete canonical face closure, the "
            "prime must be prime at most 251, and each cochain group holds "
            "at most 64 simplices. Cup products and induced maps are not "
            "computed here."
        ),
        request_type=SimplicialCohomologyRequest,
        result_type=SimplicialCohomologyResult,
        run=_run_simplicial_cohomology,
        tags=(
            "topology",
            "simplicial-cohomology",
            "betti-number",
            "cocycle-basis",
            "prime-field",
            "exact",
        ),
        discovery_terms=(
            "simplicial cohomology",
            "cochain complex",
            "cocycle basis",
            "coboundary rank",
            "cohomology basis",
            "Betti number",
        ),
        examples=(
            OperationExample(
                name="circle_cohomology_mod_two",
                description=(
                    "Compute H^0 and H^1 over F_2 for a triangle boundary; "
                    "the complex must carry its canonical face closure."
                ),
                input={
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "maximal_simplices": [["a", "b"], ["a", "c"], ["b", "c"]],
                        "faces_by_dimension": [
                            {"dimension": 0, "faces": [["a"], ["b"], ["c"]]},
                            {
                                "dimension": 1,
                                "faces": [["a", "b"], ["a", "c"], ["b", "c"]],
                            },
                        ],
                        "dimension": 1,
                        "f_vector": [3, 3],
                        "closure_size": 6,
                    },
                    "prime": 2,
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="cohomology.steenrod_square.compute",
        title="Compute Sq^0, Sq^n (cup) and Sq^k=0 for k>n over GF(2) for cocycles",
        description="Given a cochain x of degree n over GF(2) compute Sq^k(x). "
        "Supported: Sq^0(x)=x (identity), Sq^n(x)=x cup x (top, targets "
        "2n-simplices), Sq^k=0 for k>n (instability; constant work, admitted "
        "whenever the returned degree n+k stays within the declared "
        "result-degree budget); intermediate 0<k<n need cup-i and are "
        "rejected. Nonzero cochains require "
        "ambient_simplices or ambient_complex for cocycle verification; only "
        "the zero cochain is admissible without ambient. Top squares require "
        "ambient to locate targets.",
        request_type=SteenrodSquareRequest,
        result_type=SteenrodSquareResult,
        run=_run_steenrod_square,
        tags=("cohomology", "steenrod", "exact"),
        examples=(
            OperationExample(
                name="sq0_identity",
                description="Compute Sq^0(x)=x for the 1-cocycle d(vertex 0) on the triangle; nonzero cochains require ambient for cocycle verification.",
                input=_SQ_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="cohomology.bockstein.compute",
        title="Compute the bounded Bockstein homomorphism",
        description="Compute the supported exact zero Bockstein branch for a bounded cochain "
        "over Z/p. Nonzero cocycles are rejected as unsupported by this operation.",
        request_type=BocksteinRequest,
        result_type=BocksteinResult,
        run=_run_bockstein,
        tags=("cohomology", "bockstein", "exact"),
        examples=(
            OperationExample(
                name="bockstein_gf2",
                description="Compute the Bockstein of the trivial cocycle over GF(2).",
                input=_BOCKSTEIN_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.cup_product.compute",
        title="Multiply simplicial cochains by Alexander-Whitney",
        description=(
            "Multiply two prime-field simplicial cochains bound to one "
            "canonical complex by the Alexander-Whitney formula: on an "
            "ordered simplex the product evaluates the left factor on the "
            "front face and the right factor on the back face. Degrees above "
            "the complex dimension yield the empty (zero) cochain."
        ),
        request_type=CupProductRequest,
        result_type=CupProductResult,
        run=_run_cup_product,
        tags=("cohomology", "cup-product", "exact"),
        discovery_terms=(
            "cup product",
            "Alexander-Whitney",
            "simplicial cochain",
        ),
        examples=(
            OperationExample(
                name="circle_generator_square_vanishes",
                description="Square the degree-one circle generator over "
                "GF(2); degree two exceeds the circle, so the product is the "
                "empty cochain.",
                input={
                    "complex": _CIRCLE_COMPLEX,
                    "prime": 2,
                    "left": _circle_cochain([1, 0, 0]),
                    "right": _circle_cochain([1, 0, 0]),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.cohomology_ring.compute",
        title="Compute the prime-field cohomology ring table",
        description=(
            "Multiply every cohomology-basis pair within the complex "
            "dimension by Alexander-Whitney on cocycle representatives and "
            "express each product in class plus coboundary coordinates by "
            "exact row-echelon form, giving the full graded ring structure "
            "constants with the retained cohomology bases."
        ),
        request_type=CohomologyRingRequest,
        result_type=CohomologyRingResult,
        run=_run_cohomology_ring,
        tags=("cohomology", "ring", "cup-product", "exact"),
        discovery_terms=(
            "cohomology ring",
            "cup product structure",
            "cohomology multiplication",
        ),
        examples=(
            OperationExample(
                name="circle_cohomology_ring",
                description="The circle has Betti numbers (1, 1) with a "
                "trivial positive-degree product.",
                input={
                    "complex": _CIRCLE_COMPLEX,
                    "prime": 2,
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_map.induced_cohomology.compute",
        title="Pull cohomology classes back along a simplicial map",
        description=(
            "Pull target class-basis cocycles back along a simplicial vertex "
            "map with orientation signs (degenerate faces map to zero) and "
            "express each pullback in source class coordinates by exact "
            "row-echelon form. Per-degree matrices map target-class "
            "coordinates to source-class coordinates with both retained "
            "cohomology bases."
        ),
        request_type=InducedCohomologyMapRequest,
        result_type=InducedCohomologyMapResult,
        run=_run_induced_cohomology_map,
        tags=("cohomology", "simplicial-map", "induced-map", "exact"),
        discovery_terms=(
            "induced cohomology map",
            "simplicial map pullback",
            "cohomology homomorphism",
        ),
        examples=(
            OperationExample(
                name="circle_identity_induced_map",
                description="The identity map on the circle pulls every "
                "class back to itself: identity matrices.",
                input={
                    "map": {
                        "source": _CIRCLE_COMPLEX,
                        "target": _CIRCLE_COMPLEX,
                        "vertex_map": ["a", "b", "c"],
                    },
                    "prime": 2,
                    "convention": "UNREDUCED",
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
