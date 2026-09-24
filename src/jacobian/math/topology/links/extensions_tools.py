"""Public braid and Wirtinger operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.links._extensions_models import (
    AlexanderPolynomialRequest,
    AlexanderPolynomialResult,
    BraidClosureResult,
    BraidPermutationResult,
    BraidProductRequest,
    BraidWord,
    BraidWordRequest,
    ConwayPolynomialRequest,
    ConwayPolynomialResult,
    GoeritzDataRequest,
    GoeritzDataResult,
    LinkCrossingProfileRequest,
    LinkCrossingProfileResult,
    LinkDeterminantRequest,
    LinkDeterminantResult,
    LinkStateCirclesRequest,
    LinkStateCirclesResult,
    SeifertCircleRequest,
    SeifertCircleResult,
    WirtingerPresentationRequest,
    WirtingerPresentationResult,
)
from jacobian.math.topology.links.extensions import (
    braid_closure,
    braid_inverse,
    braid_multiply,
    braid_permutation,
    link_alexander_polynomial,
    link_conway_polynomial,
    link_crossing_profile,
    link_determinant,
    link_goeritz_data,
    link_seifert_circles,
    link_state_circles,
    wirtinger_presentation,
)


def _braid_permutation(request: BraidWordRequest) -> BraidPermutationResult:
    return braid_permutation(request.word)


def _braid_closure(request: BraidWordRequest) -> BraidClosureResult:
    return braid_closure(request.word)


def _braid_inverse(request: BraidWordRequest) -> BraidWord:
    return braid_inverse(request.word)


def _braid_multiply(request: BraidProductRequest) -> BraidWord:
    return braid_multiply(request.left, request.right)


def _alexander(request: AlexanderPolynomialRequest) -> AlexanderPolynomialResult:
    return link_alexander_polynomial(request.diagram)


def _conway(request: ConwayPolynomialRequest) -> ConwayPolynomialResult:
    return link_conway_polynomial(request.diagram)


def _crossing_profile(
    request: LinkCrossingProfileRequest,
) -> LinkCrossingProfileResult:
    return link_crossing_profile(request.diagram)


def _state_circles(request: LinkStateCirclesRequest) -> LinkStateCirclesResult:
    return link_state_circles(request.state)


def _goeritz(request: GoeritzDataRequest) -> GoeritzDataResult:
    return link_goeritz_data(request.diagram)


def _determinant(request: LinkDeterminantRequest) -> LinkDeterminantResult:
    return link_determinant(request.diagram)


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
        operation_id="link_diagram.state_circles.compute",
        title="Compute the circles of a complete smoothing state",
        description=(
            "Return the exact cyclic dart partition for one complete A/B smoothing "
            "state. Choices follow the diagram crossing order and the canonical "
            "counterclockwise dart convention. One state is processed in linear "
            "work; diagrams are bounded to 64 crossings and output to 8 MiB."
        ),
        request_type=LinkStateCirclesRequest,
        result_type=LinkStateCirclesResult,
        run=_state_circles,
        tags=("link-diagram", "smoothing", "state-circles", "exact"),
        discovery_terms=(
            "link smoothing state circles",
            "Kauffman state circles",
            "A/B smoothing of link diagram",
        ),
        examples=(
            OperationExample(
                name="curl_a_smoothing_circles",
                description=(
                    "Resolve the single crossing of an oriented curl using the A "
                    "smoothing and return its exact cyclic dart circle."
                ),
                input={
                    "state": {
                        "diagram": {
                            "crossings": [
                                {
                                    "crossing_id": "c0",
                                    "half_edges": ["h0", "h1", "h2", "h3"],
                                    "over_pair": [0, 2],
                                    "under_pair": [1, 3],
                                    "sign": -1,
                                }
                            ],
                            "arcs": [
                                {"tail": "h0", "head": "h3"},
                                {"tail": "h1", "head": "h2"},
                            ],
                        },
                        "choices": ["A"],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.crossing_profile.compute",
        title="Compute a link diagram's crossing profile",
        description=(
            "Return every source crossing in order with its checked sign and the "
            "ordered over/under component IDs, plus the total writhe. Equal role "
            "component IDs identify self-crossings; distinct IDs identify mixed "
            "crossings. Diagram admission allows at most 64 crossings and 128 arcs."
        ),
        request_type=LinkCrossingProfileRequest,
        result_type=LinkCrossingProfileResult,
        run=_crossing_profile,
        tags=("link-diagram", "crossing-profile", "writhe", "exact"),
        discovery_terms=("link crossing profile", "self and mixed crossings", "writhe"),
        examples=(
            OperationExample(
                name="crossing_curl_profile",
                description=(
                    "Classify the single crossing of an oriented curl; its over- and "
                    "under-passing strands belong to the same component."
                ),
                input={
                    "diagram": {
                        "crossings": [
                            {
                                "crossing_id": "c0",
                                "half_edges": ["h0", "h1", "h2", "h3"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": -1,
                            }
                        ],
                        "arcs": [
                            {"tail": "h0", "head": "h3"},
                            {"tail": "h1", "head": "h2"},
                        ],
                    }
                },
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
        operation_id="link_diagram.conway_polynomial.compute",
        title="Compute a knot diagram's Conway polynomial",
        description=(
            "Return the exact knot Conway polynomial in z, normalized by Delta(1)=1, "
            "from the source-bound Alexander polynomial using Delta(t)=nabla("
            "t^(1/2)-t^(-1/2)). Alexander symmetry and integral coefficients are "
            "required before the recurrence; this knot-only contract accepts at "
            "most eight crossings."
        ),
        request_type=ConwayPolynomialRequest,
        result_type=ConwayPolynomialResult,
        run=_conway,
        tags=("link-diagram", "knot", "Conway-polynomial", "exact"),
        discovery_terms=(
            "Conway polynomial of knot diagram",
            "Alexander-Conway polynomial",
            "Conway polynomial from Alexander polynomial",
        ),
        examples=(
            OperationExample(
                name="unknot_conway_polynomial",
                description=(
                    "The crossing-free unknot has normalized Conway polynomial 1."
                ),
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.determinant.compute",
        title="Compute a knot diagram's determinant",
        description=(
            "Return |Delta_K(-1)| as an exact nonnegative integer, together with "
            "the source-bound normalized Alexander polynomial and its signed "
            "evaluation. This knot-only operation inherits the eight-crossing "
            "exact Alexander bound; it does not infer link determinant conventions."
        ),
        request_type=LinkDeterminantRequest,
        result_type=LinkDeterminantResult,
        run=_determinant,
        tags=("link-diagram", "knot", "determinant", "exact"),
        discovery_terms=("knot determinant", "determinant from Alexander polynomial"),
        examples=(
            OperationExample(
                name="unknot_determinant",
                description="The crossing-free unknot has determinant 1.",
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
                                    "crossing_000:dart_3",
                                    "crossing_000:dart_2",
                                    "crossing_000:dart_1",
                                ],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": 1,
                            },
                            {
                                "crossing_id": "crossing_001",
                                "half_edges": [
                                    "crossing_001:dart_0",
                                    "crossing_001:dart_3",
                                    "crossing_001:dart_2",
                                    "crossing_001:dart_1",
                                ],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": 1,
                            },
                        ],
                        "arcs": [
                            {
                                "tail": "crossing_001:dart_3",
                                "head": "crossing_000:dart_0",
                            },
                            {
                                "tail": "crossing_001:dart_2",
                                "head": "crossing_000:dart_1",
                            },
                            {
                                "tail": "crossing_000:dart_2",
                                "head": "crossing_001:dart_1",
                            },
                            {
                                "tail": "crossing_000:dart_3",
                                "head": "crossing_001:dart_0",
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
            "Use the encoded component orientation to apply "
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
        operation_id="braid.word.multiply.compute",
        title="Multiply braid words in one braid group",
        description=(
            "Concatenate two signed Artin presentation words in B_n, requiring "
            "the same explicit strand count. The 64-letter result bound is "
            "checked before construction; no braid-word reduction is performed."
        ),
        request_type=BraidProductRequest,
        result_type=BraidWord,
        run=_braid_multiply,
        tags=("braid", "group-operation", "word", "exact"),
        discovery_terms=("multiply braid words", "braid group product"),
        examples=(
            OperationExample(
                name="braid_word_product",
                description="Concatenate sigma_1 and its inverse in B_2.",
                input={
                    "left": {
                        "strand_count": 2,
                        "letters": [{"generator": 1, "exponent": 1}],
                    },
                    "right": {
                        "strand_count": 2,
                        "letters": [{"generator": 1, "exponent": -1}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="braid.word.inverse.compute",
        title="Invert a braid word",
        description=(
            "Return the inverse presentation word in the same B_n by reversing "
            "letter order and negating each exponent. Strand count is retained "
            "exactly; the operation is linear in the bounded word length."
        ),
        request_type=BraidWordRequest,
        result_type=BraidWord,
        run=_braid_inverse,
        tags=("braid", "group-operation", "inverse", "exact"),
        discovery_terms=("inverse braid word", "braid group inverse"),
        examples=(
            OperationExample(
                name="inverse_trefoil_braid_word",
                description="Invert sigma_1 cubed in B_2.",
                input={"word": _SIGMA_ONE_CUBED},
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
