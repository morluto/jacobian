# ruff: noqa: F403,F405
"""Public declarations for finite simplicial topology transforms."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.operations import canonicalize
from jacobian.math.topology.release import *


def _run_poset(r: FacePosetRequest) -> FacePosetResult:
    return face_poset(r)


def _run_clique(r: CliqueRequest) -> CliqueResult:
    return clique_complex(canonicalize(r.complex.vertices, r.complex.facets).complex)


def _run_graph_clique(r: GraphCliqueRequest) -> CliqueResult:
    return graph_clique_complex(r.graph)


def _run_orient(r: OrientabilityRequest) -> OrientabilityResult:
    return orientability(r)


def _run_local(r: LocalHomologyRequest) -> LocalHomologyResult:
    return local_homology(r)


def _run_manifold(r: HomologyManifoldRequest) -> HomologyManifoldResult:
    return homology_manifold(r)


_TRIANGLE = {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]}
TOOLS = (
    MathTool(
        operation_id="topology.simplicial_complex.face_poset.compute",
        title="Compute the face poset and order complex",
        description="Return every nonempty face, strict inclusion relation, and the order complex of a finite simplicial complex; chains are represented by source face indices.",
        request_type=FacePosetRequest,
        result_type=FacePosetResult,
        run=_run_poset,
        tags=("topology", "simplicial", "face-poset", "order-complex", "exact"),
        examples=(
            OperationExample(
                name="triangle_face_poset",
                description="Compute the face poset of one filled triangle; faces are closed and ordered by inclusion.",
                input={"complex": _TRIANGLE},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.clique.compute",
        title="Reconstruct the clique complex of a one-skeleton",
        description="Extract the one-skeleton of a finite simplicial complex and return the flag/clique complex reconstructed from exactly its graph edges.",
        request_type=CliqueRequest,
        result_type=CliqueResult,
        run=_run_clique,
        tags=("topology", "simplicial", "clique", "flag", "exact"),
        examples=(
            OperationExample(
                name="triangle_clique",
                description="Reconstruct the clique complex of a triangle graph; every pair of vertices is an edge.",
                input={"complex": _TRIANGLE},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.clique_from_graph.compute",
        title="Compute a graph's clique complex",
        description="Return the exact flag complex of a bounded indexed simple graph, including isolated graph vertices as singleton simplices.",
        request_type=GraphCliqueRequest,
        result_type=CliqueResult,
        run=_run_graph_clique,
        tags=("topology", "simplicial", "graph", "clique", "flag", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_clique",
                description="The clique complex of a four-cycle is the four-edge cycle because it has no triangles.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [0, 3], [1, 2], [2, 3]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.orientability.compute",
        title="Compute facet-sign orientability",
        description="Propagate coherent signs across codimension-one facet adjacency and return an exact orientability decision with the obstructing ridge when signs conflict.",
        request_type=OrientabilityRequest,
        result_type=OrientabilityResult,
        run=_run_orient,
        tags=("topology", "simplicial", "orientability", "exact"),
        examples=(
            OperationExample(
                name="triangle_orientability",
                description="Orient the filled triangle; its single facet has a coherent sign.",
                input={"complex": _TRIANGLE},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.local_homology.compute",
        title="Compute reduced homology of one simplicial link",
        description="Compute reduced finite-field homology of the exact link of a selected face, retaining the source complex, face, link, coefficient prime, and degenerate empty-link case.",
        request_type=LocalHomologyRequest,
        result_type=LocalHomologyResult,
        run=_run_local,
        tags=("topology", "simplicial", "local-homology", "link", "exact"),
        examples=(
            OperationExample(
                name="triangle_vertex_link",
                description="Compute the reduced link homology of vertex a in a filled triangle; the selected simplex must be a face.",
                input={"complex": _TRIANGLE, "simplex": ["a"], "prime": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.homology_manifold.decide",
        title="Decide the bounded simplicial homology-manifold condition",
        description="Check every nonempty face link against the sphere-link homology condition over a selected prime field; a failing face is retained as the mathematical obstruction.",
        request_type=HomologyManifoldRequest,
        result_type=HomologyManifoldResult,
        run=_run_manifold,
        tags=("topology", "simplicial", "homology-manifold", "exact"),
        examples=(
            OperationExample(
                name="circle_manifold",
                description="Check a three-edge circle as a one-dimensional homology manifold; every face link has sphere homology.",
                input={
                    "complex": {
                        "vertices": ["a", "b", "c"],
                        "facets": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "prime": 2,
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
