"""Public declaration for exact link-diagram component traversal."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.links._models import (
    LinkBracketRequest,
    LinkBracketResult,
    LinkComponentsRequest,
    LinkComponentsResult,
    LinkDiagramMirrorRequest,
    LinkDiagramMirrorResult,
    LinkingMatrixResult,
    LinkJonesRequest,
    LinkJonesResult,
    LinkOrientationReverseRequest,
    LinkOrientationReverseResult,
)
from jacobian.math.topology.links.extensions_tools import TOOLS as EXTENSION_TOOLS
from jacobian.math.topology.links.operations import (
    link_bracket,
    link_components,
    link_jones,
    link_linking_matrix,
    link_mirror,
    link_orientation_reverse,
)


def _run_link_components(request: LinkComponentsRequest) -> LinkComponentsResult:
    return link_components(request.diagram)


def _run_bracket(request: LinkBracketRequest) -> LinkBracketResult:
    return link_bracket(request.diagram)


def _run_jones(request: LinkJonesRequest) -> LinkJonesResult:
    return link_jones(request.diagram)


def _run_linking(request: LinkComponentsRequest) -> LinkingMatrixResult:
    return link_linking_matrix(request.diagram)


def _run_mirror(request: LinkDiagramMirrorRequest) -> LinkDiagramMirrorResult:
    return link_mirror(request.diagram)


def _run_orientation_reverse(
    request: LinkOrientationReverseRequest,
) -> LinkOrientationReverseResult:
    return link_orientation_reverse(request.diagram, request.component_representatives)


TOOLS = (
    *EXTENSION_TOOLS,
    MathTool(
        operation_id="link_diagram.orientation_reverse.compute",
        title="Reverse selected link component orientations",
        description=(
            "Reverse selected crossing-bearing components by source crossing-dart "
            "representatives. Preserve crossing and dart identities, return complete "
            "component/dart transport, and report exactly the crossing signs that "
            "change. Crossing-free loops lack stable IDs in this representation and "
            "cannot be selected."
        ),
        request_type=LinkOrientationReverseRequest,
        result_type=LinkOrientationReverseResult,
        run=_run_orientation_reverse,
        tags=("link-diagram", "orientation", "exact"),
        discovery_terms=("reverse link component orientation", "orient a link diagram"),
        examples=(
            OperationExample(
                name="reverse_one_hopf_component",
                description=(
                    "Reverse one component of a two-component Hopf diagram; both "
                    "mixed crossing signs reverse."
                ),
                input={
                    "diagram": {
                        "crossings": [
                            {
                                "crossing_id": "c0",
                                "half_edges": ["a0", "b0", "a1", "b1"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": -1,
                            },
                            {
                                "crossing_id": "c1",
                                "half_edges": ["a2", "b2", "a3", "b3"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": -1,
                            },
                        ],
                        "arcs": [
                            {"tail": "a1", "head": "b2"},
                            {"tail": "a3", "head": "b0"},
                            {"tail": "b1", "head": "a2"},
                            {"tail": "b3", "head": "a0"},
                        ],
                    },
                    "component_representatives": ["a0"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.mirror.compute",
        title="Mirror a classical oriented link diagram",
        description=(
            "Exchange over- and under-passing strands at every crossing and "
            "negate each oriented crossing sign. Crossing IDs, half-edge IDs, "
            "arc pairings, and free-loop count are retained exactly; this "
            "constructs a mirror diagram and does not test link equivalence."
        ),
        request_type=LinkDiagramMirrorRequest,
        result_type=LinkDiagramMirrorResult,
        run=_run_mirror,
        tags=("link-diagram", "mirror", "exact"),
        discovery_terms=("mirror a knot diagram", "mirror a link diagram"),
        examples=(
            OperationExample(
                name="mirror_crossing_curl",
                description=(
                    "Mirror a one-crossing oriented curl by exchanging the "
                    "over/under pairs and negating the crossing sign."
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
        operation_id="link_diagram.components.compute",
        title="Partition a classical oriented link diagram into components",
        description=(
            "For a well-formed classical oriented link diagram with "
            "counterclockwise crossing rotations, checked signs, and directed "
            "tail/head arcs in a sphere embedding, return the complete component "
            "partition in encoded orientation: "
            "cyclic dart sequences with per-crossing OVER/UNDER roles and "
            "component lengths. Every half-edge lies in exactly one crossing "
            "and one directed arc; traversal gives disjoint oriented cycles "
            "exactly once."
        ),
        request_type=LinkComponentsRequest,
        result_type=LinkComponentsResult,
        run=_run_link_components,
        tags=("link-diagram", "components", "exact"),
        discovery_terms=(
            "link diagram components",
            "knot diagram traversal",
            "crossing over under roles",
        ),
        examples=(
            OperationExample(
                name="hopf_link_two_components",
                description=(
                    "Partition the two-crossing Hopf link into two components; "
                    "every half-edge must lie in exactly one crossing and one "
                    "arc."
                ),
                input={
                    "diagram": {
                        "crossings": [
                            {
                                "crossing_id": "c0",
                                "half_edges": ["a0", "b0", "a1", "b1"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": -1,
                            },
                            {
                                "crossing_id": "c1",
                                "half_edges": ["a2", "b2", "a3", "b3"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                                "sign": -1,
                            },
                        ],
                        "arcs": [
                            {"tail": "a1", "head": "b2"},
                            {"tail": "a3", "head": "b0"},
                            {"tail": "b1", "head": "a2"},
                            {"tail": "b3", "head": "a0"},
                        ],
                        "free_loops": 0,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.bracket.compute",
        title="Compute the exact Kauffman bracket state sum",
        description=(
            "Enumerate every A/B smoothing state of an oriented link diagram and "
            "return its exact Laurent bracket with complete state-circle data; "
            "the exact state-sum envelope admits at most 12 crossings (2^12 states)."
        ),
        request_type=LinkBracketRequest,
        result_type=LinkBracketResult,
        run=_run_bracket,
        tags=("link-diagram", "kauffman-bracket", "laurent", "exact"),
        discovery_terms=("Kauffman bracket", "link smoothing", "state circles"),
        examples=(
            OperationExample(
                name="unknot_bracket",
                description="Compute the bracket of one free unknot; a single component has bracket 1.",
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.jones.compute",
        title="Compute the exact Jones polynomial",
        description=(
            "Return V_D in the A-variable convention V_D(t)=(-A)^(-3w(D)) "
            "<D>(A), with t=A^(-4); the returned exact Laurent polynomial uses "
            "variable A, so no square-root variable is implicit for links. The "
            "complete state sum admits at most 12 crossings (2^12 states), "
            "80,000,000 conservative state-sum work units, and a 4 MiB "
            "result-size bound, even though the diagram carrier permits 64 crossings."
        ),
        request_type=LinkJonesRequest,
        result_type=LinkJonesResult,
        run=_run_jones,
        tags=("link-diagram", "jones", "laurent", "exact"),
        discovery_terms=(
            "Jones polynomial",
            "writhe normalization",
            "ordinary Jones variable t",
            "knot invariant",
        ),
        examples=(
            OperationExample(
                name="unknot_jones",
                description="Compute the Jones polynomial of one free unknot; the normalized Laurent value is 1.",
                input={"diagram": {"free_loops": 1}},
            ),
        ),
    ),
    MathTool(
        operation_id="link_diagram.linking_matrix.compute",
        title="Compute the exact pairwise linking matrix",
        description=(
            "Sum signed mixed crossings and divide by two to return the source-bound "
            "symmetric linking matrix; component traversal admits at most 64 crossings."
        ),
        request_type=LinkComponentsRequest,
        result_type=LinkingMatrixResult,
        run=_run_linking,
        tags=("link-diagram", "linking-number", "exact"),
        discovery_terms=("linking number", "linking matrix", "Hopf linking"),
        examples=(
            OperationExample(
                name="unlink_linking",
                description="Compute the linking matrix of two free unknots; distinct components have linking number zero.",
                input={"diagram": {"free_loops": 2}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
