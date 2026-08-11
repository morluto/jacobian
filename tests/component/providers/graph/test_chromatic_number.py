"""Z3-backed exact chromatic-number capability contracts."""

from __future__ import annotations

import pytest
import z3  # type: ignore[import-untyped]
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus


def _invoke(
    services: DomainTestServices,
    graph: dict[str, object],
) -> CapabilityResult:
    return services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.chromatic_number.compute",
            input={"graph": graph, "resource_budget": {"wall_seconds": 5}},
        )
    )


def test_chromatic_number_returns_first_satisfying_k_with_witness(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services
    result = _invoke(
        runtime,
        {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"], ["c", "a"]],
        },
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    output = result.output["result"]
    assert output["status"] == "EXACT"
    assert output["chromatic_number"] == 3
    assert output["lower_bound"] == 3
    assert output["upper_bound"] == 3
    assert [step["status"] for step in output["tested"]] == [
        "UNSATISFIABLE",
        "SATISFIABLE",
    ]
    assert result.artifact_uris == ()
    assert result.obligations == ()
    assert result.relationships == ()

    coloring = output["coloring"]
    assert set(coloring) == {"a", "b", "c"}
    assert all(
        coloring[left] != coloring[right]
        for left, right in (
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
        )
    )


def test_chromatic_number_timeout_returns_unknown_result_without_artifacts(
    graph_optimization_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = graph_optimization_services
    monkeypatch.setattr(z3.Solver, "check", lambda _solver: z3.unknown)

    result = _invoke(
        runtime,
        {
            "vertices": ["a", "b", "c", "d", "e"],
            "edges": [
                ["a", "b"],
                ["b", "c"],
                ["c", "d"],
                ["d", "e"],
                ["e", "a"],
            ],
        },
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    output = result.output["result"]
    assert output["status"] == "UNKNOWN"
    assert output["solver_status"] == "UNKNOWN"
    assert output["tested"] == [{"colors": 2, "status": "UNKNOWN"}]
    assert result.artifact_uris == ()


def test_chromatic_number_rejects_repeated_undirected_edges(
    graph_optimization_services: DomainTestServices,
) -> None:
    runtime = graph_optimization_services
    result = _invoke(
        runtime,
        {
            "vertices": ["a", "b"],
            "edges": [["a", "b"], ["b", "a"]],
        },
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.diagnostics[0].code == "INVALID_CHROMATIC_NUMBER_REQUEST"
    assert result.artifact_uris == ()


def test_chromatic_number_rejects_result_for_a_different_vertex_universe(
    graph_optimization_services: DomainTestServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = graph_optimization_services
    from jacobian.domains.graph_optimization import chromatic_number

    original = chromatic_number.solve_chromatic_number

    def invalid_result(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.model_copy(
            update={
                "vertices": ("missing",),
                "order": 1,
                "coloring": {"missing": 0},
            }
        )

    monkeypatch.setattr(chromatic_number, "solve_chromatic_number", invalid_result)
    result = _invoke(
        runtime,
        {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        },
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CHROMATIC_NUMBER_COLORING_INVALID"
    assert result.artifact_uris == ()
