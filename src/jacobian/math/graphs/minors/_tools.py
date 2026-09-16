"""Deterministic H-minor-model checker declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.minors._models import (
    MinorModelCheckRequest,
    MinorModelCheckResult,
    MinorModelFindRequest,
    MinorModelFindResult,
    TopologicalMinorCheckRequest,
    TopologicalMinorCheckResult,
    TopologicalMinorFindRequest,
    TopologicalMinorFindResult,
)
from jacobian.math.graphs.minors.operations import (
    check_minor_model,
    check_topological_minor,
    find_minor_model,
    find_topological_minor,
)


def _check(request: MinorModelCheckRequest) -> MinorModelCheckResult:
    return check_minor_model(
        request.source, request.target, request.branch_sets, request.witnesses
    )


def _find_minor_model(request: MinorModelFindRequest) -> MinorModelFindResult:
    return find_minor_model(request.source, request.target, request.resource_budget)


def _check_topological_minor(
    request: TopologicalMinorCheckRequest,
) -> TopologicalMinorCheckResult:
    return check_topological_minor(
        request.source, request.target, request.branch_vertices, request.paths
    )


def _find_topological_minor(
    request: TopologicalMinorFindRequest,
) -> TopologicalMinorFindResult:
    return find_topological_minor(
        request.source, request.target, request.resource_budget
    )


TOOLS = (
    MathTool(
        operation_id="graph.minor_model.check",
        title="Check an explicit H-minor model in G",
        description=(
            "Deterministically check one candidate H-minor model: every target "
            "vertex needs exactly one nonempty branch set, branch sets are "
            "pairwise disjoint, each induces a connected source subgraph, and "
            "every target edge needs a source-edge witness crossing its two "
            "branch sets. VALID_MINOR_MODEL returns canonical branch sets, "
            "per-branch connectivity ledgers, the witness ledger, and used "
            "versus deleted source structure; INVALID_MINOR_MODEL returns the "
            "first concrete obstruction. Extra source edges between branch "
            "sets stay legal for ordinary minors."
        ),
        request_type=MinorModelCheckRequest,
        result_type=MinorModelCheckResult,
        run=_check,
        tags=("graph", "minor", "branch-set", "checker", "exact"),
        discovery_terms=("minor model", "branch sets", "graph minor"),
        examples=(
            OperationExample(
                name="edge_in_a_triangle",
                description="A single target edge modelled by two singleton branch sets in a triangle.",
                input={
                    "source": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                    "branch_sets": [
                        {"target": "x", "members": ["a"]},
                        {"target": "y", "members": ["b"]},
                    ],
                    "witnesses": [{"targets": ["x", "y"], "source_edge": ["a", "b"]}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.minor_model.find",
        title="Search for an H-minor model in G",
        description=(
            "Bounded backtracking search deciding whether target H is a minor "
            "of source G. Every source vertex is assigned to one target branch "
            "or to deletion; the first assignment with nonempty connected "
            "branches and a crossing source edge per target edge is replayed "
            "through the minor-model checker. FOUND returns the checked branch "
            "sets and edge witnesses, EXHAUSTED certifies the complete "
            "assignment space holds no model (with the enumerated-candidate "
            "receipt), and UNKNOWN reports candidate-budget truncation, which "
            "is never an EXHAUSTED verdict."
        ),
        request_type=MinorModelFindRequest,
        result_type=MinorModelFindResult,
        run=_find_minor_model,
        tags=("graph", "minor", "branch-set", "bounded-search", "exact"),
        discovery_terms=("minor model", "branch sets", "graph minor", "minor test"),
        examples=(
            OperationExample(
                name="triangle_minor_of_wheel",
                description="A triangle minor in the 4-vertex wheel: rim branch "
                "sets plus the hub, with rim edges as witnesses.",
                input={
                    "source": {
                        "vertices": ["hub", "a", "b", "c"],
                        "edges": [
                            ["a", "b"],
                            ["a", "hub"],
                            ["b", "c"],
                            ["b", "hub"],
                            ["a", "c"],
                            ["c", "hub"],
                        ],
                    },
                    "target": {
                        "vertices": ["x", "y", "z"],
                        "edges": [["x", "y"], ["x", "z"], ["y", "z"]],
                    },
                    "resource_budget": {"max_candidates": 50000},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.topological_minor.check",
        title="Check an explicit subdivision model of H in G",
        description=(
            "Deterministically check one candidate subdivision model: every "
            "target vertex needs exactly one branch vertex, branch vertices "
            "are pairwise distinct, every target edge needs a source-vertex "
            "path running from the branch vertex of its first endpoint to the "
            "branch vertex of its second, each path is simple with every step "
            "a source edge, and path interiors are disjoint from all branch "
            "vertices and from every other path interior. VALID_SUBDIVISION "
            "returns the canonical branch map, path ledger, and used versus "
            "deleted source structure; INVALID_SUBDIVISION returns the first "
            "concrete obstruction."
        ),
        request_type=TopologicalMinorCheckRequest,
        result_type=TopologicalMinorCheckResult,
        run=_check_topological_minor,
        tags=("graph", "topological-minor", "subdivision", "checker", "exact"),
        discovery_terms=(
            "topological minor",
            "subdivision",
            "disjoint paths",
            "branch vertices",
        ),
        examples=(
            OperationExample(
                name="subdivided_edge_in_a_path",
                description="A single target edge modelled by the full length-3 "
                "path through a four-vertex source path.",
                input={
                    "source": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                    },
                    "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                    "branch_vertices": [
                        {"target": "x", "source": "a"},
                        {"target": "y", "source": "d"},
                    ],
                    "paths": [
                        {"targets": ["x", "y"], "vertices": ["a", "b", "c", "d"]}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.topological_minor.find",
        title="Search for a subdivision of H in G",
        description=(
            "Bounded backtracking search deciding whether source G holds a "
            "subdivision of target H. Injective branch-vertex maps range over "
            "the source vertices and each target edge is routed through "
            "internally vertex-disjoint source paths; the first complete "
            "routing is replayed through the subdivision checker. FOUND "
            "returns the checked branch vertices and path witnesses, "
            "EXHAUSTED certifies the complete search space holds no "
            "subdivision, and UNKNOWN reports probe-budget truncation, which "
            "is never an EXHAUSTED verdict."
        ),
        request_type=TopologicalMinorFindRequest,
        result_type=TopologicalMinorFindResult,
        run=_find_topological_minor,
        tags=(
            "graph",
            "topological-minor",
            "subdivision",
            "bounded-search",
            "exact",
        ),
        discovery_terms=(
            "topological minor",
            "subdivision",
            "disjoint paths",
            "minor test",
        ),
        examples=(
            OperationExample(
                name="triangle_subdivision_in_wheel",
                description="A triangle subdivision in the 4-vertex wheel with "
                "singleton branch vertices on the rim and hub.",
                input={
                    "source": {
                        "vertices": ["hub", "a", "b", "c"],
                        "edges": [
                            ["a", "b"],
                            ["a", "hub"],
                            ["b", "c"],
                            ["b", "hub"],
                            ["a", "c"],
                            ["c", "hub"],
                        ],
                    },
                    "target": {
                        "vertices": ["x", "y", "z"],
                        "edges": [["x", "y"], ["x", "z"], ["y", "z"]],
                    },
                    "resource_budget": {"max_candidates": 50000},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
