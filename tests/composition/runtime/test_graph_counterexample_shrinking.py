from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.support.state import copy_template

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.plugin_execution import PluginExecutionResult
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime


@pytest.fixture
def redundant_odd_cycle_runtime(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[tuple[JacobianRuntime, str]]:
    with _open_runtime_with_redundant_odd_cycle(
        tmp_path,
        template=authorized_portfolio_template,
    ) as opened:
        yield opened


def test_graph_counterexample_shrink_records_verified_steps_and_exact_local_scope(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime

    result = _shrink(runtime, graph_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert runtime.core.store.get(result.output["final_graph_uri"]).payload == {
        "graph_schema_version": "1",
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
    }
    accepted = [
        attempt
        for attempt in result.output["attempts"]
        if attempt["outcome"] == "ACCEPTED_VERIFIED"
    ]
    assert [attempt["deleted_vertex"] for attempt in accepted] == ["d"]
    assert all(
        attempt["verification_record_uri"] in result.artifact_uris
        for attempt in accepted
    )
    scope = result.output["local_minimality_scope"]
    assert scope["tested_vertex_deletions"] == ["a", "b", "c"]
    assert scope["tested_edge_deletions"] == [
        ["a", "b"],
        ["a", "c"],
        ["b", "c"],
    ]
    assert scope["complete_for_requested_reducers"] is True
    assert scope["one_step_locally_minimal"] is True
    assert scope["global_minimality_claimed"] is False
    assert scope["expected_attempt_count"] == 6
    assert scope["completed_attempt_count"] == 6
    assert scope["completeness_status"] == "COMPLETE"
    assert scope["remaining_obligations"] == []
    assert all(
        attempt["candidate_digest"] is not None
        for attempt in result.output["attempts"]
        if attempt["proposed_graph_uri"] is not None
    )
    trace = runtime.core.store.get(result.output["trace_uri"])
    assert trace.payload["attempts"] == result.output["attempts"]
    assert trace.payload["local_minimality_scope"] == scope


def test_graph_counterexample_shrink_budget_reports_only_tested_scope(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime

    result = _shrink(runtime, graph_uri, evaluation_budget=2)

    scope = result.output["local_minimality_scope"]
    assert scope["complete_for_requested_reducers"] is False
    assert scope["one_step_locally_minimal"] is False
    assert scope["global_minimality_claimed"] is False
    assert scope["untested_vertex_deletions"] or scope["untested_edge_deletions"]


def test_graph_counterexample_shrink_timeout_returns_incumbent_without_minimality(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime
    runtime.services.shrinking.executor = _TimeoutExecutor()  # type: ignore[assignment]

    result = _shrink(runtime, graph_uri)

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["final_graph_uri"] == graph_uri
    assert result.output["attempts"] == []
    assert result.output["local_minimality_scope"]["one_step_locally_minimal"] is False
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC


def test_graph_counterexample_shrink_requires_compatible_registered_checker(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime
    incompatible = runtime.portfolio.graph.degree_sequence_checker_id
    assert incompatible is not None

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.counterexample.shrink",
            input={
                "graph_uri": graph_uri,
                "property_id": "graph.property.non_bipartite",
                "property_checker_id": incompatible,
                "reducers": ["delete_vertex", "delete_edge"],
                "evaluation_budget": 20,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "GRAPH_PROPERTY_CHECKER_INVALID"


def test_graph_counterexample_shrink_fails_closed_on_tampered_graph(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime
    graph = runtime.core.store.get(graph_uri)
    runtime.core.store._blob_path(graph.manifest.payload_digest).write_bytes(
        b"tampered"
    )

    result = _shrink(runtime, graph_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["code"] == "GRAPH_SHRINK_INPUT_INVALID"
    assert result.diagnostics[0].code == "GRAPH_SHRINK_INPUT_INVALID"


def test_graph_counterexample_shrink_rejects_unrelated_reducer_edits(
    redundant_odd_cycle_runtime: tuple[JacobianRuntime, str],
) -> None:
    runtime, graph_uri = redundant_odd_cycle_runtime
    runtime.services.shrinking.executor = _UnrelatedEditExecutor()  # type: ignore[assignment]

    result = _shrink(runtime, graph_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["final_graph_uri"] == graph_uri
    assert result.output["attempts"][0]["outcome"] == "INVALID_REDUCTION"
    assert result.output["attempts"][0]["verification_record_uri"] is None
    assert "exact single-vertex deletion" in result.output["attempts"][0]["detail"]


def test_graph_counterexample_shrink_order_is_deterministic(
    tmp_path: Path, authorized_portfolio_template: Path
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    with (
        _open_runtime_with_redundant_odd_cycle(
            first_root, template=authorized_portfolio_template
        ) as (first, first_graph),
        _open_runtime_with_redundant_odd_cycle(
            second_root, template=authorized_portfolio_template
        ) as (second, second_graph),
    ):
        first_result = _shrink(first, first_graph)
        second_result = _shrink(second, second_graph)

        def signature(output: dict[str, Any]) -> list[tuple[Any, ...]]:
            return [
                (
                    attempt["reducer"],
                    attempt["deleted_vertex"],
                    attempt["deleted_edge"],
                    attempt["outcome"],
                )
                for attempt in output["attempts"]
            ]

        assert signature(first_result.output) == signature(second_result.output)
        assert (
            first.core.store.get(first_result.output["final_graph_uri"]).payload
            == second.core.store.get(second_result.output["final_graph_uri"]).payload
        )


class _TimeoutExecutor:
    def run(self, **_: Any) -> PluginExecutionResult:
        return PluginExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            output=None,
            diagnostics="",
            detail="fixture reducer timeout",
            runtime_ms=1,
        )


class _UnrelatedEditExecutor:
    def run(self, **_: Any) -> PluginExecutionResult:
        return PluginExecutionResult(
            status=ExecutionStatus.COMPLETED,
            output={
                "response_version": "1",
                "current_objectives": {"vertices": 4, "edges": 4},
                "reductions": [
                    {
                        "reducer": "delete_vertex",
                        "payload": {
                            "graph_schema_version": "1",
                            "vertices": ["a", "b", "c", "e"],
                            "edges": [
                                ["a", "b"],
                                ["a", "c"],
                                ["b", "c"],
                            ],
                        },
                        "objectives": {"vertices": 3, "edges": 3},
                    }
                ],
            },
            diagnostics="",
            detail=None,
            runtime_ms=1,
        )


@contextmanager
def _open_runtime_with_redundant_odd_cycle(
    root: Path,
    *,
    template: Path,
) -> Iterator[tuple[JacobianRuntime, str]]:
    root = copy_template(template, root / "state")
    with create_runtime(
        root, checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED
    ) as runtime:
        graph = runtime.core.artifacts.put(
            schema_uri=runtime.portfolio.graph.graph_schema_uri,
            semantics_uri=runtime.portfolio.graph.semantics_uri,
            payload={
                "graph_schema_version": "1",
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]],
            },
            summary="non-bipartite graph with one redundant leaf",
        )
        yield runtime, graph.artifact_uri


def _shrink(
    runtime: JacobianRuntime,
    graph_uri: str,
    *,
    evaluation_budget: int = 20,
) -> Any:
    checker_id = runtime.portfolio.graph_shrinking.property_checker_id
    assert checker_id is not None
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.counterexample.shrink",
            input={
                "graph_uri": graph_uri,
                "property_id": "graph.property.non_bipartite",
                "property_checker_id": checker_id,
                "reducers": ["delete_vertex", "delete_edge"],
                "evaluation_budget": evaluation_budget,
            },
        )
    )
