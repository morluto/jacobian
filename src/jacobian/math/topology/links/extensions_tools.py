"""Public braid and Wirtinger operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.links._extensions_models import (
    AlexanderPolynomialRequest,
    AlexanderPolynomialResult,
    BraidArtinActionResult,
    BraidClosureResult,
    BraidPermutationResult,
    BraidWordRequest,
    GoeritzDataRequest,
    GoeritzDataResult,
    SeifertCircleRequest,
    SeifertCircleResult,
    WirtingerPresentationRequest,
    WirtingerPresentationResult,
)
from jacobian.math.topology.links.extensions import (
    braid_artin_action,
    braid_closure,
    braid_permutation,
    link_alexander_polynomial,
    link_goeritz_data,
    link_seifert_circles,
    wirtinger_presentation,
)


def _braid_permutation(request: BraidWordRequest) -> BraidPermutationResult:
    return braid_permutation(request.word)


def _braid_artin_action(request: BraidWordRequest) -> BraidArtinActionResult:
    return braid_artin_action(request.word)


def _braid_closure(request: BraidWordRequest) -> BraidClosureResult:
    return braid_closure(request.word)


def _alexander(request: AlexanderPolynomialRequest) -> AlexanderPolynomialResult:
    return link_alexander_polynomial(request.diagram)


def _goeritz(request: GoeritzDataRequest) -> GoeritzDataResult:
    return link_goeritz_data(request.diagram)


def _seifert(request: SeifertCircleRequest) -> SeifertCircleResult:
    return link_seifert_circles(request.diagram)


def _wirtinger(
    request: WirtingerPresentationRequest,
) -> WirtingerPresentationResult:
    return wirtinger_presentation(request.diagram)


_SIGMA_ONE_CUBED = {
    "strand_count": 2,
    "letters": [
        {"generator": 1, "exponent": 1},
        {"generator": 1, "exponent": 1},
        {"generator": 1, "exponent": 1},
    ],
}


TOOLS: MathTools = (
    MathTool(
        operation_id="braid.word.artin_action.compute",
        title="Apply a braid word to its strand free group",
        description=(
            "Return the exact Artin automorphism of the free group on the braid "
            "strands. The declared convention sends sigma_i to x_i -> "
            "x_i*x_(i+1)*x_i^-1 and x_(i+1) -> x_i; negative letters use its "
            "inverse, and letters act successively from left to right. Images are "
            "freely reduced. The admitted output is limited to 128 letters per "
            "image and 100000 cumulative substitution letters."
        ),
        request_type=BraidWordRequest,
        result_type=BraidArtinActionResult,
        run=_braid_artin_action,
        tags=("braid", "Artin-action", "free-group", "exact"),
        discovery_terms=(
            "braid action on free group",
            "Artin representation of braid group",
            "automorphism induced by braid word",
        ),
        examples=(
            OperationExample(
                name="trefoil_braid_artin_action",
                description=(
                    "Apply sigma_1 cubed to the rank-two free group; the positive "
                    "generator convention and its threefold iterate are explicit."
                ),
                input={"word": _SIGMA_ONE_CUBED},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.alexander_polynomial.compute",
        title="Compute a knot diagram's Alexander polynomial",
        description=(
            "Build the exact Fox matrix of the canonical Wirtinger presentation, "
            "take a codimension-one determinant, and return the primitive "
            "one-variable Alexander polynomial normalized to nonnegative exponents "
            "and positive constant term. This bounded contract accepts one-"
            "component diagrams with at most eight crossings."
        ),
        request_type=AlexanderPolynomialRequest,
        result_type=AlexanderPolynomialResult,
        run=_alexander,
        tags=("link-diagram", "knot", "Alexander-polynomial", "exact"),
        discovery_terms=(
            "Alexander polynomial of knot diagram",
            "Fox calculus knot invariant",
        ),
        examples=(
            OperationExample(
                name="unknot_alexander_polynomial",
                description=(
                    "Compute Delta(t)=1 for the crossing-free unknot; the input "
                    "must have exactly one component."
                ),
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.goeritz_matrix.compute",
        title="Construct a link diagram's Goeritz matrix",
        description=(
            "Enumerate the planar projection regions, choose the checkerboard "
            "shading containing the least boundary dart, assign incidence +1 "
            "when the shaded corners are the overpassing pair and -1 otherwise, "
            "and delete the last shaded region from the signed Laplacian. Return "
            "that exact reduced integral Goeritz matrix and its absolute "
            "determinant. This slice requires a connected "
            "nonempty projection with at most 32 crossings; it does not claim a "
            "signature correction."
        ),
        request_type=GoeritzDataRequest,
        result_type=GoeritzDataResult,
        run=_goeritz,
        tags=("link-diagram", "Goeritz-matrix", "checkerboard", "exact"),
        discovery_terms=(
            "Goeritz matrix of link diagram",
            "checkerboard regions signed Tait graph",
        ),
        examples=(
            OperationExample(
                name="positive_hopf_goeritz_matrix",
                description=(
                    "Construct the one-by-one Goeritz matrix of the positive "
                    "two-crossing braid closure; the projection must be planar and "
                    "connected."
                ),
                input={
                    "diagram": {
                        "crossings": [
                            {
                                "crossing_id": "crossing_000",
                                "half_edges": [
                                    "crossing_000:dart_0",
                                    "crossing_000:dart_1",
                                    "crossing_000:dart_2",
                                    "crossing_000:dart_3",
                                ],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": 1,
                            },
                            {
                                "crossing_id": "crossing_001",
                                "half_edges": [
                                    "crossing_001:dart_0",
                                    "crossing_001:dart_1",
                                    "crossing_001:dart_2",
                                    "crossing_001:dart_3",
                                ],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": 1,
                            },
                        ],
                        "arcs": [
                            {
                                "first": "crossing_001:dart_3",
                                "second": "crossing_000:dart_0",
                            },
                            {
                                "first": "crossing_001:dart_2",
                                "second": "crossing_000:dart_1",
                            },
                            {
                                "first": "crossing_000:dart_2",
                                "second": "crossing_001:dart_1",
                            },
                            {
                                "first": "crossing_000:dart_3",
                                "second": "crossing_001:dart_0",
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.seifert_circles.compute",
        title="Construct a knot diagram's Seifert circles",
        description=(
            "Choose the canonical direction of the unique link component, apply "
            "the oriented smoothing at every crossing, and return every Seifert "
            "circle with source darts plus the disk-band surface Euler "
            "characteristic and genus. The genus belongs to this constructed "
            "surface and is not a minimum-genus claim."
        ),
        request_type=SeifertCircleRequest,
        result_type=SeifertCircleResult,
        run=_seifert,
        tags=("link-diagram", "knot", "Seifert-circles", "surface"),
        discovery_terms=(
            "Seifert smoothing circles",
            "Seifert algorithm surface genus",
        ),
        examples=(
            OperationExample(
                name="unknot_seifert_circle",
                description=(
                    "Construct the single Seifert disk of a crossing-free unknot; "
                    "this knot-first operation requires one component."
                ),
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="braid.word.permutation.compute",
        title="Compute a braid word's strand permutation",
        description=(
            "Apply every signed Artin letter to the labelled strand axis and "
            "return the exact permutation, its cycle partition, closure component "
            "count, and signed exponent sum. Generator signs affect writhe but not "
            "the strand transposition."
        ),
        request_type=BraidWordRequest,
        result_type=BraidPermutationResult,
        run=_braid_permutation,
        tags=("braid", "permutation", "closure-components", "exact"),
        discovery_terms=(
            "braid strand permutation",
            "braid closure component count",
            "Artin word exponent sum",
        ),
        examples=(
            OperationExample(
                name="trefoil_braid_permutation",
                description=(
                    "Compute the strand transposition and one closure cycle of "
                    "sigma_1 cubed in B_2; every generator index must be smaller "
                    "than the strand count."
                ),
                input={"word": _SIGMA_ONE_CUBED},
            ),
        ),
    ),
    MathTool(
        operation_id="braid.word.closure.compute",
        title="Construct the standard closure of a braid word",
        description=(
            "Convert a bounded signed Artin word into one exact classical oriented "
            "link diagram with a crossing per letter, explicit closure arcs, and "
            "the source strand permutation. This constructs the standard closure; "
            "it does not decide braid or link equivalence."
        ),
        request_type=BraidWordRequest,
        result_type=BraidClosureResult,
        run=_braid_closure,
        tags=("braid", "link-diagram", "closure", "exact"),
        discovery_terms=(
            "braid closure link diagram",
            "Artin braid to knot diagram",
        ),
        examples=(
            OperationExample(
                name="trefoil_braid_closure",
                description=(
                    "Construct the three-crossing standard closure of sigma_1 "
                    "cubed in B_2; the 64-letter braid envelope bounds the complete "
                    "diagram output."
                ),
                input={"word": _SIGMA_ONE_CUBED},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.wirtinger_presentation.compute",
        title="Construct a link diagram's Wirtinger presentation",
        description=(
            "Return one finite group presentation with a generator for each "
            "canonical overpass class, one source-bound conjugation relator per "
            "crossing, and explicit half-edge transport. Positive crossings use "
            "o u_in o^-1 u_out^-1 and negative crossings invert the conjugating "
            "over-generator. The result is a group "
            "presentation, not a word-problem or link-equivalence conclusion."
        ),
        request_type=WirtingerPresentationRequest,
        result_type=WirtingerPresentationResult,
        run=_wirtinger,
        tags=("link-diagram", "Wirtinger", "group-presentation", "exact"),
        discovery_terms=(
            "link group presentation",
            "Wirtinger presentation",
            "knot group generators relators",
        ),
        examples=(
            OperationExample(
                name="unknot_wirtinger_presentation",
                description=(
                    "Construct the one-generator free presentation of a zero-"
                    "crossing unknot; every free loop contributes one meridian."
                ),
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
