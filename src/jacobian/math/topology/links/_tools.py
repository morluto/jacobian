"""Public declaration for exact link-diagram component traversal."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.links._models import (
    LinkBracketRequest,
    LinkBracketResult,
    LinkComponentsRequest,
    LinkComponentsResult,
    LinkingMatrixResult,
    LinkJonesRequest,
    LinkJonesResult,
)
from jacobian.math.topology.links.extensions_tools import TOOLS as EXTENSION_TOOLS
from jacobian.math.topology.links.operations import (
    link_bracket,
    link_components,
    link_jones,
    link_linking_matrix,
)


def _run_link_components(request: LinkComponentsRequest) -> LinkComponentsResult:
    return link_components(request.diagram)


def _run_bracket(request: LinkBracketRequest) -> LinkBracketResult:
    return link_bracket(request.diagram)


def _run_jones(request: LinkJonesRequest) -> LinkJonesResult:
    return link_jones(request.diagram)


def _run_linking(request: LinkComponentsRequest) -> LinkingMatrixResult:
    return link_linking_matrix(request.diagram)


TOOLS = (
    *EXTENSION_TOOLS,
    MathTool(
        operation_id="link_diagram.components.compute",
        title="Partition a classical oriented link diagram into components",
        description=(
            "For a well-formed classical oriented link diagram with "
            "dart/half-edge crossings carrying over/under pairs and an arc "
            "involution, return the complete oriented component partition: "
            "cyclic dart sequences with per-crossing OVER/UNDER roles and "
            "component lengths. Every half-edge lies in exactly one crossing "
            "and one arc; traversal gives disjoint cycles covering every arc "
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
                            },
                            {
                                "crossing_id": "c1",
                                "half_edges": ["a2", "b2", "a3", "b3"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                            },
                        ],
                        "arcs": [
                            {"first": "a1", "second": "b2"},
                            {"first": "a3", "second": "b0"},
                            {"first": "b1", "second": "a2"},
                            {"first": "b3", "second": "a0"},
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
        title="Compute the exact writhe-normalized Jones polynomial",
        description=(
            "Compute the Kauffman bracket and apply A^-3w normalization using the "
            "diagram's oriented crossing signs; exact bracket work admits at most "
            "12 crossings even though the diagram carrier permits 64."
        ),
        request_type=LinkJonesRequest,
        result_type=LinkJonesResult,
        run=_run_jones,
        tags=("link-diagram", "jones", "laurent", "exact"),
        discovery_terms=("Jones polynomial", "writhe normalization", "knot invariant"),
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
