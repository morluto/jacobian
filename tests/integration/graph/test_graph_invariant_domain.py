from __future__ import annotations

from pathlib import Path

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.kernel import JacobianKernel


def _graph(
    vertices: list[str],
    edges: list[list[str]],
) -> dict[str, object]:
    return {"vertices": vertices, "edges": edges}


def test_graph_invariant_family_boundaries_and_witnesses(tmp_path: Path) -> None:
    kernel = JacobianKernel(tmp_path)
    triangle_tail = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["b", "c"], ["a", "c"], ["c", "d"]],
    )
    cases = (
        (
            "graph.invariant.girth.compute",
            triangle_tail,
            {"girth": 3, "has_cycle": True},
        ),
        (
            "graph.invariant.diameter.compute",
            triangle_tail,
            {
                "status": "COMPUTED",
                "diameter": 2,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
        (
            "graph.invariant.edge_connectivity.compute",
            triangle_tail,
            {"edge_connectivity": 1},
        ),
        (
            "graph.invariant.vertex_connectivity.compute",
            triangle_tail,
            {"vertex_connectivity": 1},
        ),
        (
            "graph.invariant.is_eulerian.compute",
            triangle_tail,
            {"is_eulerian": False},
        ),
        (
            "graph.invariant.spanning_tree_count.compute",
            triangle_tail,
            {"spanning_tree_count": 3, "connected": True},
        ),
    )
    for capability_id, graph, expected in cases:
        result = kernel.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input={"graph": graph})
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.output["result"] == expected
        assert len(result.artifact_uris) == 2

    matching = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={"graph": triangle_tail},
        )
    )
    assert matching.output["result"] == {
        "maximum_matching_cardinality": 2,
        "witness_edges": [["a", "b"], ["c", "d"]],
        "certificate": {
            "certificate_schema_version": "1",
            "kind": "TUTTE_BERGE_BARRIER",
            "barrier_vertices": [],
            "odd_component_count": 0,
            "upper_bound": 2,
        },
    }
    assert matching.capability_version == "2"

    star = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={
                "graph": _graph(
                    ["center", "x", "y", "z"],
                    [["center", "x"], ["center", "y"], ["center", "z"]],
                )
            },
        )
    )
    assert star.output["result"]["maximum_matching_cardinality"] == 1
    assert star.output["result"]["certificate"] == {
        "certificate_schema_version": "1",
        "kind": "TUTTE_BERGE_BARRIER",
        "barrier_vertices": ["center"],
        "odd_component_count": 3,
        "upper_bound": 1,
    }


def test_disconnected_and_acyclic_graph_conventions(tmp_path: Path) -> None:
    kernel = JacobianKernel(tmp_path)
    graph = _graph(["a", "b", "c"], [["a", "b"]])
    diameter = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.diameter.compute",
            input={"graph": graph},
        )
    )
    girth = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.girth.compute",
            input={"graph": graph},
        )
    )
    trees = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.spanning_tree_count.compute",
            input={"graph": graph},
        )
    )
    assert diameter.execution.status is ExecutionStatus.COMPLETED
    assert diameter.output["result"] == {
        "status": "NOT_APPLICABLE",
        "diameter": None,
        "connected": False,
        "exactness": "NOT_APPLICABLE",
        "detail": "diameter requires a nonempty connected graph",
    }
    assert girth.output["result"] == {"girth": 0, "has_cycle": False}
    assert trees.output["result"] == {
        "spanning_tree_count": 0,
        "connected": False,
    }


def test_radius_uses_explicit_not_applicable_for_disconnected_graph(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path)

    connected = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.radius.compute",
            input={"graph": _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])},
        )
    )
    disconnected = kernel.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.radius.compute",
            input={"graph": _graph(["a", "b", "c"], [["a", "b"]])},
        )
    )

    assert connected.capability_version == "2"
    assert disconnected.capability_version == "2"
    assert connected.output["result"] == {
        "status": "COMPUTED",
        "radius": 1,
        "connected": True,
        "exactness": "EXACT",
        "detail": None,
    }
    assert disconnected.execution.status is ExecutionStatus.COMPLETED
    assert disconnected.output["result"] == {
        "status": "NOT_APPLICABLE",
        "radius": None,
        "connected": False,
        "exactness": "NOT_APPLICABLE",
        "detail": "radius requires a nonempty connected graph",
    }


def test_np_hard_invariants_are_budgeted_and_carry_obligations(
    tmp_path: Path,
) -> None:
    kernel = JacobianKernel(tmp_path)
    cycle = _graph(
        ["a", "b", "c", "d", "e"],
        [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["a", "e"]],
    )
    for capability_id, optimum in (
        ("graph.invariant.clique_number.compute", 2),
        ("graph.invariant.independence_number.compute", 2),
    ):
        result = kernel.capabilities.invoke(
            CapabilityRequest(
                capability_id=capability_id,
                input={
                    "graph": cycle,
                    "resource_budget": {
                        "wall_seconds": 5,
                        "max_solver_calls": 33,
                        "max_order": 32,
                    },
                },
            )
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.output["status"] == "EXACT"
        assert result.output["optimum_value"] == optimum
        assert len(result.output["witness_vertices"]) == optimum
        assert len(result.artifact_uris) == 3
        obligation = kernel.store.get(result.obligations[0].obligation_uri)
        assert obligation.payload["claimed_value"] == optimum
