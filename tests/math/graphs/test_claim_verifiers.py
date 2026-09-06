from __future__ import annotations

import json

from jacobian.math.graphs.coloring import (
    verify_k_colorability,
    verify_maximal_independent_set,
)
from jacobian.math.graphs.coloring._models import (
    KColorabilityResult,
    MaximalIndependentSetResult,
    VertexColoringAssignment,
)
from jacobian.math.graphs.coloring.equitable_k_coloring import (
    decide_equitable_k_coloring,
    verify_equitable_coloring,
)
from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
    EquitableColoringResult,
)
from jacobian.math.graphs.multigraph import (
    multigraph_flow_check,
    verify_multigraph_flow_check,
)
from jacobian.math.graphs.multigraph._models import (
    FiniteAbelianGroup,
    LooplessMultigraph,
    MultigraphEdge,
    MultigraphFlowCheckRequest,
    MultigraphFlowCheckResult,
)
from jacobian.math.graphs.symmetry import (
    graph_symmetry_orbits,
    verify_graph_symmetry_orbits,
)
from jacobian.math.graphs.symmetry._models import GraphSymmetryOrbitRequest
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)


def test_vertex_coloring_verifier_rejects_forged_serialized_relation() -> None:
    graph = IndexedSimpleUndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2)))
    assignment = VertexColoringAssignment(graph=graph, colors=2, coloring=(0, 0, 1))
    claim = KColorabilityResult.model_validate(
        {
            "graph": graph.model_dump(),
            "colors": 2,
            "colorable": True,
            "vertex_count": 3,
            "coloring": assignment.model_dump(),
        }
    )
    assert not verify_k_colorability(
        KColorabilityResult.model_validate_json(claim.model_dump_json())
    )


def test_maximal_independent_set_verifier_checks_obstruction_relation() -> None:
    graph = IndexedSimpleUndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2)))
    claim = MaximalIndependentSetResult.model_validate(
        {
            "graph": graph.model_dump(),
            "candidate_set": [0],
            "decision": "INDEPENDENT_NOT_MAXIMAL",
            "addable_vertex": 1,
        }
    )
    assert not verify_maximal_independent_set(claim)


def test_equitable_verifier_rejects_forged_serialized_relation() -> None:
    graph = SimpleUndirectedGraph(
        vertices=("a", "b", "c", "d"),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    result = decide_equitable_k_coloring(graph, 2)
    payload = result.model_dump()
    payload["coloring"]["coloring"] = (0, 0, 1, 1)
    forged = EquitableColoringResult.model_validate(payload)
    assert not verify_equitable_coloring(
        EquitableColoringResult.model_validate_json(forged.model_dump_json())
    )


def test_symmetry_verifier_rejects_arbitrary_complete_partition() -> None:
    request = GraphSymmetryOrbitRequest.model_validate(
        {
            "graph": {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "vertex_colors": ["endpoint", "middle", "endpoint"],
            },
            "generators": [
                {
                    "generator_id": "reflection",
                    "mapping": [["a", "c"], ["b", "b"], ["c", "a"]],
                }
            ],
        }
    )
    result = graph_symmetry_orbits(request.graph, request.generators)
    payload = result.model_dump()
    payload["vertex_orbits"] = [
        {"orbit_index": 0, "representative": "a", "members": ["a"]},
        {"orbit_index": 1, "representative": "b", "members": ["b", "c"]},
    ]
    forged = type(result).model_validate(payload)
    assert not verify_graph_symmetry_orbits(
        type(result).model_validate_json(forged.model_dump_json())
    )


def test_multigraph_flow_checker_diagnoses_invalid_candidate_and_verifies_roundtrip() -> None:
    graph = LooplessMultigraph(
        vertex_count=2,
        edges=(
            MultigraphEdge(edge_id="e0", left=0, right=1),
            MultigraphEdge(edge_id="e1", left=0, right=1),
        ),
    )
    group = FiniteAbelianGroup(moduli=(3,))
    request = MultigraphFlowCheckRequest.model_validate_json(
        json.dumps(
            {
                "graph": graph.model_dump(),
                "group": group.model_dump(),
                "edge_values": [
                    {
                        "edge_id": "e0",
                        "orientation": "left_to_right",
                        "value": [3],
                    }
                ],
            }
        )
    )
    result = multigraph_flow_check(graph, group, request.edge_values)
    assert not result.assignment_valid
    assert {diagnostic.code for diagnostic in result.assignment_diagnostics} == {
        "MISSING_EDGE",
        "RESIDUE_OUT_OF_RANGE",
    }
    assert verify_multigraph_flow_check(
        MultigraphFlowCheckResult.model_validate_json(result.model_dump_json())
    )
