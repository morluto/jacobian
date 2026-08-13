from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import build_graph_optimization_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_graph_optimization_bundle(),
    ) as services:
        yield services


def _biggs_smith_graph() -> dict[str, object]:
    vertices = [f"{index}{letter}" for index in range(1, 18) for letter in "abcdef"]
    edges: set[tuple[str, str]] = set()

    def add_edge(left: str, right: str) -> None:
        edges.add(tuple(sorted((left, right))))

    for index in range(1, 18):
        for leaf in "ab":
            add_edge(f"{index}{leaf}", f"{index}e")
        for leaf in "cd":
            add_edge(f"{index}{leaf}", f"{index}f")
        add_edge(f"{index}e", f"{index}f")
    for letter, step in (("a", 1), ("b", 4), ("c", 2), ("d", 8)):
        for index in range(1, 18):
            target = ((index - 1 + step) % 17) + 1
            add_edge(f"{index}{letter}", f"{target}{letter}")
    return {
        "vertices": vertices,
        "edges": [list(edge) for edge in sorted(edges)],
    }


def test_independence_number_reproduces_biggs_smith_alpha_43(
    domain_services: DomainTestServices,
) -> None:
    graph = _biggs_smith_graph()
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.independence_number.compute",
            input={
                "graph": graph,
                "resource_budget": {"wall_seconds": 120, "max_order": 128},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    value = result.output["result"]
    assert value["status"] == "EXACT"
    assert value["order"] == 102
    assert value["optimum_value"] == 43
    assert value["lower_bound"] == value["upper_bound"] == 43
    witness = set(value["witness_vertices"])
    assert len(witness) == 43
    assert all(
        left not in witness or right not in witness for left, right in graph["edges"]
    )
    assert result.artifact_uris == ()


def test_independence_number_rejects_order_above_128_without_artifacts(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.independence_number.compute",
            input={
                "graph": {
                    "vertices": [f"v{index:03d}" for index in range(129)],
                    "edges": [],
                },
                "resource_budget": {"wall_seconds": 5, "max_order": 128},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_GRAPH_INDEPENDENCE_NUMBER_REQUEST"
    assert result.artifact_uris == ()
