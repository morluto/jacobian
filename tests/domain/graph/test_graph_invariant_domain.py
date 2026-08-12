from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import (
    build_graph_invariant_bundle,
    build_graph_optimization_bundle,
)
from jacobian.graphs import atlas_search, invariants
from jacobian.graphs.artifacts import nx


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_graph_invariant_bundle(),
        build_graph_optimization_bundle(),
    ) as services:
        yield services


def _graph(
    vertices: list[str],
    edges: list[list[str]],
) -> dict[str, object]:
    return {"vertices": vertices, "edges": edges}


_TRIANGLE_TAIL = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [["a", "b"], ["b", "c"], ["a", "c"], ["c", "d"]],
}


def test_graph_invariant_family_boundaries_and_witnesses(
    domain_services,
) -> None:
    cases = [
        ("graph.invariant.girth.compute", {"girth": 3, "has_cycle": True}),
        (
            "graph.invariant.diameter.compute",
            {
                "status": "COMPUTED",
                "diameter": 2,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
        ("graph.invariant.edge_connectivity.compute", {"edge_connectivity": 1}),
        ("graph.invariant.vertex_connectivity.compute", {"vertex_connectivity": 1}),
        ("graph.invariant.is_eulerian.compute", {"is_eulerian": False}),
        (
            "graph.invariant.spanning_tree_count.compute",
            {"spanning_tree_count": 3, "connected": True},
        ),
    ]
    for capability_id, expected in cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=capability_id, input={"graph": _TRIANGLE_TAIL}
            )
        )
        assert result.execution.status is ExecutionStatus.COMPLETED, capability_id
        assert result.output["result"] == expected, capability_id
        assert result.artifact_uris == (), capability_id


def test_maximum_matching_and_star_conventions(domain_services) -> None:
    triangle_tail = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["b", "c"], ["a", "c"], ["c", "d"]],
    )

    matching = domain_services.core.capabilities.invoke(
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
    assert matching.capability_version == "3"

    star = domain_services.core.capabilities.invoke(
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


def test_maximum_matching_has_a_capability_specific_64_vertex_bound(
    domain_services,
) -> None:
    vertices = [f"v{index:02d}" for index in range(64)]
    edges = [[vertices[index], vertices[index + 1]] for index in range(0, 64, 2)]

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={"graph": _graph(vertices, edges)},
        )
    )

    assert result.output["result"]["maximum_matching_cardinality"] == 32
    assert result.output["result"]["witness_edges"] == edges
    assert result.output["result"]["certificate"] == {
        "certificate_schema_version": "1",
        "kind": "TUTTE_BERGE_BARRIER",
        "barrier_vertices": [],
        "odd_component_count": 0,
        "upper_bound": 32,
    }

    too_large = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.maximum_matching.compute",
            input={"graph": _graph([*vertices, "v64"], edges)},
        )
    )
    assert too_large.execution.status is ExecutionStatus.ERROR
    assert too_large.diagnostics[0].code == "INVALID_GRAPH_MAXIMUM_MATCHING_REQUEST"

    unrelated_invariant = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.diameter.compute",
            input={"graph": _graph(vertices[:33], edges[:16])},
        )
    )
    assert unrelated_invariant.execution.status is ExecutionStatus.ERROR
    assert unrelated_invariant.diagnostics[0].code == "INVALID_GRAPH_INVARIANT_REQUEST"


def test_disconnected_and_acyclic_graph_conventions(domain_services) -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"]])
    diameter = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.diameter.compute",
            input={"graph": graph},
        )
    )
    girth = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.girth.compute",
            input={"graph": graph},
        )
    )
    trees = domain_services.core.capabilities.invoke(
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
    domain_services,
) -> None:

    connected = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.radius.compute",
            input={"graph": _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])},
        )
    )
    disconnected = domain_services.core.capabilities.invoke(
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


_CYCLE_5 = {
    "vertices": ["a", "b", "c", "d", "e"],
    "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["a", "e"]],
}


class _InconsistentCliqueBackend:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def max_weight_clique(
        self, *_args: object, **_kwargs: object
    ) -> tuple[set[object], int]:
        return set(), 1


@pytest.mark.parametrize(
    "compute",
    (
        lambda graph: invariants._independence_number_property(
            graph, "independence_number"
        ),
        atlas_search._compute_all_properties,
    ),
)
def test_graph_backends_with_inconsistent_independence_results_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    compute,
) -> None:
    backend = _InconsistentCliqueBackend(nx())
    monkeypatch.setattr(invariants, "nx", lambda: backend)
    monkeypatch.setattr(atlas_search, "nx", lambda: backend)

    with pytest.raises(CapabilityInvocationError) as caught:
        compute(nx().path_graph(2))

    assert caught.value.diagnostic.code == "INCONSISTENT_INDEPENDENCE_RESULT"


def test_np_hard_invariants_return_exact_typed_results(
    domain_services,
) -> None:
    cases = [
        ("graph.invariant.clique_number.compute", 2),
        ("graph.invariant.independence_number.compute", 2),
    ]
    for capability_id, optimum in cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=capability_id,
                input={
                    "graph": _CYCLE_5,
                    "resource_budget": {
                        "wall_seconds": 5,
                        "max_solver_calls": 33,
                        "max_order": 32,
                    },
                },
            )
        )
        assert result.execution.status is ExecutionStatus.COMPLETED, capability_id
        output = result.output["result"]
        assert output["status"] == "EXACT", capability_id
        assert output["optimum_value"] == optimum, capability_id
        assert len(output["witness_vertices"]) == optimum, capability_id
        assert result.artifact_uris == (), capability_id


def test_graph_invariant_resource_atomics_are_exact_computed(
    domain_services: DomainTestServices,
) -> None:
    cases = (
        (
            "graph.invariant.triangle_count.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                }
            },
            {"triangle_count": 1},
        ),
        (
            "graph.invariant.radius.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            },
            {
                "status": "COMPUTED",
                "radius": 1,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
        (
            "graph.invariant.radius.compute",
            {"graph": {"vertices": [], "edges": []}},
            {
                "status": "NOT_APPLICABLE",
                "radius": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "radius requires a nonempty connected graph",
            },
        ),
        (
            "graph.k_core.compute",
            {
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]],
                },
                "k": 2,
            },
            {"k": 2, "vertices": ["a", "b", "c"]},
        ),
    )
    for capability_id, payload, expected in cases:
        result = domain_services.core.capabilities.invoke(
            CapabilityRequest(capability_id=capability_id, input=payload)
        )
        assert result.execution.status is ExecutionStatus.COMPLETED
        assert result.output["result"] == expected
