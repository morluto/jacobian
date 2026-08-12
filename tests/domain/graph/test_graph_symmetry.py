from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_symmetry import build_graph_symmetry_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_graph_symmetry_bundle(),
    ) as services:
        yield services


def _result_payload(services: DomainTestServices, result: object) -> dict[str, object]:
    del services
    return result.output["result"]  # type: ignore[attr-defined]


def test_cycle_rotation_has_one_vertex_and_one_edge_orbit(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["a", "b"],
                        ["a", "d"],
                        ["b", "c"],
                        ["c", "d"],
                    ],
                },
                "generators": [
                    {
                        "generator_id": "quarter_turn",
                        "mapping": {
                            "a": "b",
                            "b": "c",
                            "c": "d",
                            "d": "a",
                        },
                    }
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    computed = _result_payload(domain_services, result)
    assert computed["vertex_orbits"] == [
        {
            "orbit_index": 0,
            "representative": "a",
            "members": ["a", "b", "c", "d"],
        }
    ]
    assert computed["edge_orbits"] == [
        {
            "orbit_index": 0,
            "representative": ["a", "b"],
            "members": [
                ["a", "b"],
                ["a", "d"],
                ["b", "c"],
                ["c", "d"],
            ],
        }
    ]
    assert computed["orbit_completeness"] == "COMPLETE_FOR_DECLARED_GENERATORS"
    assert (
        computed["automorphism_group_completeness"]
        == "FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"
    )
    assert result.artifact_uris == ()


def test_colored_path_reflection_preserves_declared_classes(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "generators": [
                    {
                        "generator_id": "reflection",
                        "mapping": {"a": "c", "b": "b", "c": "a"},
                    }
                ],
                "vertex_colors": [
                    {"vertex": "a", "color": "terminal-class"},
                    {"vertex": "b", "color": "middle"},
                    {"vertex": "c", "color": "terminal-class"},
                ],
                "edge_colors": [
                    {"edge": ["a", "b"], "color": "p=1/2"},
                    {"edge": ["b", "c"], "color": "p=1/2"},
                ],
            },
        )
    )

    computed = _result_payload(domain_services, result)
    assert [orbit["members"] for orbit in computed["vertex_orbits"]] == [
        ["a", "c"],
        ["b"],
    ]
    assert [orbit["members"] for orbit in computed["edge_orbits"]] == [
        [["a", "b"], ["b", "c"]]
    ]
    assert computed["vertex_color_mode"] == "DECLARED"
    assert computed["edge_color_mode"] == "DECLARED"


def test_empty_generator_set_materializes_identity_subgroup_orbits(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {
                    "vertices": ["a", "b"],
                    "edges": [["a", "b"]],
                },
                "generators": [],
            },
        )
    )

    computed = _result_payload(domain_services, result)
    assert computed["generator_count"] == 0
    assert [orbit["members"] for orbit in computed["vertex_orbits"]] == [["a"], ["b"]]
    assert [orbit["members"] for orbit in computed["edge_orbits"]] == [[["a", "b"]]]


def test_color_breaking_generator_fails_before_artifact_writes(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
                "generators": [
                    {
                        "generator_id": "reflection",
                        "mapping": {"a": "c", "b": "b", "c": "a"},
                    }
                ],
                "vertex_colors": [
                    {"vertex": "a", "color": "source"},
                    {"vertex": "b", "color": "middle"},
                    {"vertex": "c", "color": "target"},
                ],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_GRAPH_SYMMETRY_REQUEST"


def test_maximum_contract_payloads_fit_artifact_and_checker_budgets(
    domain_services: DomainTestServices,
) -> None:
    vertices = [f"v{index:03d}" + "x" * 60 for index in range(256)]
    edges = [
        [vertices[left], vertices[right]]
        for left in range(256)
        for right in range(left + 1, 256)
    ][:4_096]
    identity = {vertex: vertex for vertex in vertices}
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {"vertices": vertices, "edges": edges},
                "generators": [
                    {
                        "generator_id": f"g{index:02d}" + "x" * 61,
                        "mapping": identity,
                    }
                    for index in range(64)
                ],
                "vertex_colors": [
                    {"vertex": vertex, "color": "c" * 128} for vertex in vertices
                ],
                "edge_colors": [{"edge": edge, "color": "c" * 128} for edge in edges],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.artifact_uris == ()
    output_bytes = len(canonicalize_json(result.output["result"]))
    assert output_bytes < 10 * 1024 * 1024


def test_multibyte_payload_over_artifact_budget_is_a_scored_input_error(
    domain_services: DomainTestServices,
) -> None:
    vertices = [f"v{index:03d}" + "🧮" * 60 for index in range(256)]
    edges = [
        [vertices[left], vertices[right]]
        for left in range(256)
        for right in range(left + 1, 256)
    ][:4_096]
    identity = {vertex: vertex for vertex in vertices}

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.symmetry.generator_orbits.compute",
            input={
                "graph": {"vertices": vertices, "edges": edges},
                "generators": [
                    {
                        "generator_id": f"g{index:02d}" + "🧮" * 61,
                        "mapping": identity,
                    }
                    for index in range(64)
                ],
                "vertex_colors": [
                    {"vertex": vertex, "color": "🧮" * 128} for vertex in vertices
                ],
                "edge_colors": [{"edge": edge, "color": "🧮" * 128} for edge in edges],
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "REQUEST_RESOURCE_LIMIT_EXCEEDED"
    assert result.artifact_uris == ()
