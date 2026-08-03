"""Tests for knowledge.search capability: filter semantics and completeness reporting."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.memory import ResearchEpisode
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime.model import JacobianRuntime


def test_knowledge_search_filters_episode_domain_tags_and_failures(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    graph_episode = ResearchEpisode(
        capability_id="graph.compute.properties",
        capability_version="1",
        mode=CapabilityMode.EXPLORE,
        request={"graph": "K5"},
        result={
            "output": {"failure_classifications": ["nonplanar_obstruction"]},
            "diagnostics": [],
        },
        assurance_level=CapabilityAssuranceLevel.COMPUTED,
        summary="K5 counterexample with a nonplanar obstruction",
        tags=("graph", "counterexample", "failure"),
    )
    graph_uri = runtime.core.memory.record(graph_episode)
    runtime.core.memory.record(
        ResearchEpisode(
            capability_id="lean.check",
            capability_version="1",
            mode=CapabilityMode.VERIFY,
            request={"statement": "True"},
            result={"output": {"conclusion": "TRUE"}, "diagnostics": []},
            assurance_level=CapabilityAssuranceLevel.VERIFIED,
            summary="Lean replay succeeded",
            tags=("lean", "proof"),
        )
    )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="knowledge.search",
            input={
                "query": "counterexample",
                "domains": ["graph"],
                "tags_all": ["failure"],
                "tags_any": ["counterexample", "proof"],
                "failure_stages": ["mathematical_evaluation"],
                "failure_classifications": ["nonplanar_obstruction"],
                "limit": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.capability_version == "2"
    assert result.completeness.status.value == "COMPLETE"
    assert result.scope is not None
    assert [hit["episode_uri"] for hit in result.output["hits"]] == [graph_uri]
    assert result.output["hits"][0]["matched_query_terms"] == ["counterexample"]
    assert result.output["hits"][0]["matched_filters"] == [
        "domains",
        "tags_all",
        "tags_any",
        "failure_stages",
        "failure_classifications",
    ]
    assert result.output["indexed_episode_count"] == 2
    assert result.scope.parameters["index_snapshot"] == result.output["index_snapshot"]
    assert result.output["total_matches"] == 1
    assert result.output["returned_count"] == 1
    assert result.output["truncated"] is False
    assert result.output["completeness"] == "COMPLETE"
    assert result.output["index_snapshot"].startswith("sha256:")


def test_knowledge_search_reports_snapshot_bounded_partial_results(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    for value in (1, 2):
        runtime.core.memory.record(
            ResearchEpisode(
                capability_id="polynomial.factor.compute",
                capability_version="1",
                mode=CapabilityMode.EXPLORE,
                request={"value": value},
                result={"output": {"value": value}, "diagnostics": []},
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
                summary=f"factor episode {value}",
                tags=("polynomial",),
            )
        )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="knowledge.search",
            input={"domains": ["polynomial"], "limit": 1},
        )
    )

    assert result.output["indexed_episode_count"] == 2
    assert result.completeness.status.value == "PARTIAL"
    assert result.scope is not None
    assert result.scope.parameters["index_snapshot"] == result.output["index_snapshot"]
    assert result.output["total_matches"] == 2
    assert result.output["returned_count"] == 1
    assert result.output["truncated"] is True
    assert result.output["completeness"] == "PARTIAL"
