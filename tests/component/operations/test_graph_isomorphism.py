from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs.isomorphism import (
    GraphIsomorphismResources,
    build_graph_isomorphism_operation,
)
from jacobian.graphs.operation_resources import (
    GraphOperationResources,
    build_graph_operations,
)


@dataclass(frozen=True, slots=True)
class GraphIsomorphismTestServices(DomainTestServices):
    graph: GraphOperationResources
    isomorphism: GraphIsomorphismResources


@contextmanager
def _open_graph_isomorphism_services(
    root: Path,
    *,
    authorize_checker: bool,
) -> Iterator[GraphIsomorphismTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            graph_adapters, graph = build_graph_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            for adapter in graph_adapters:
                services.installation.register_operation(adapter)
            adapter, isomorphism = build_graph_isomorphism_operation(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                graph,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_operation(adapter)
        yield GraphIsomorphismTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            graph=graph,
            isomorphism=isomorphism,
        )


@pytest.fixture
def graph_isomorphism_services(
    tmp_path: Path,
) -> Iterator[GraphIsomorphismTestServices]:
    with _open_graph_isomorphism_services(
        tmp_path / "state", authorize_checker=True
    ) as services:
        yield services


@pytest.fixture
def unauthorized_graph_isomorphism_services(
    tmp_path: Path,
) -> Iterator[GraphIsomorphismTestServices]:
    with _open_graph_isomorphism_services(
        tmp_path / "state", authorize_checker=False
    ) as services:
        yield services


def _graph_uri(
    runtime: GraphIsomorphismTestServices,
    *,
    vertices: list[str],
    edges: list[list[str]],
) -> str:
    return runtime.core.artifacts.put(
        schema_uri=runtime.graph.graph_schema_uri,
        semantics_uri=runtime.graph.semantics_uri,
        payload={
            "graph_schema_version": "1",
            "vertices": vertices,
            "edges": edges,
        },
        summary="graph-isomorphism test input",
    ).artifact_uri


def _input(
    runtime: GraphIsomorphismTestServices,
    mapping: dict[str, str],
) -> dict[str, object]:
    left_graph_uri = _graph_uri(
        runtime,
        vertices=["a", "b", "c"],
        edges=[["a", "b"], ["b", "c"]],
    )
    right_graph_uri = _graph_uri(
        runtime,
        vertices=["x", "y", "z"],
        edges=[["x", "z"], ["y", "z"]],
    )
    return {
        "left_graph_uri": left_graph_uri,
        "right_graph_uri": right_graph_uri,
        "mapping": mapping,
    }


def test_graph_isomorphism_verifies_a_valid_bijection(
    graph_isomorphism_services,
) -> None:

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input=_input(graph_isomorphism_services, {"a": "x", "b": "z", "c": "y"}),
        )
    )

    assert result.output["is_isomorphism"] is True
    assert result.output["conclusion"] == "TRUE"
    assert result.output["coverage"] == "EXHAUSTIVE"
    assert result.verification_record_uri is not None
    assert result.output["verification_record_uri"] in result.artifact_uris


def test_graph_isomorphism_verifies_a_negative_result(
    graph_isomorphism_services,
) -> None:

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input=_input(graph_isomorphism_services, {"a": "x", "b": "y", "c": "z"}),
        )
    )

    assert result.output["is_isomorphism"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.verification_record_uri is not None
    assert result.output["first_violation"] == {
        "kind": "ADJACENCY_MISMATCH",
        "source_vertices": ["a", "b"],
        "mapped_vertices": ["x", "y"],
        "source_adjacent": True,
        "target_adjacent": False,
        "vertex": None,
        "mapped_vertex": None,
    }


def test_graph_isomorphism_preserves_an_empty_missing_vertex_label(
    graph_isomorphism_services,
) -> None:
    left_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["", "a"],
        edges=[],
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["x", "y"],
        edges=[],
    )

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": left_graph_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {"a": "x"},
            },
        )
    )

    assert result.output["conclusion"] == "FALSE"
    assert result.output["first_violation"]["kind"] == "SOURCE_DOMAIN_MISMATCH"
    assert result.output["first_violation"]["vertex"] == ""


def test_graph_isomorphism_reports_missing_target_for_empty_source(
    graph_isomorphism_services,
) -> None:
    left_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=[],
        edges=[],
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["target"],
        edges=[],
    )

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": left_graph_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {},
            },
        )
    )

    assert result.output["conclusion"] == "FALSE"
    assert result.output["first_violation"] == {
        "kind": "TARGET_BIJECTION_MISMATCH",
        "source_vertices": None,
        "mapped_vertices": None,
        "source_adjacent": None,
        "target_adjacent": None,
        "vertex": None,
        "mapped_vertex": "target",
    }


def test_graph_isomorphism_keeps_checker_rejection_unknown(
    graph_isomorphism_services,
) -> None:
    checker_id = graph_isomorphism_services.isomorphism.checker_id
    assert checker_id is not None
    request_input = _input(graph_isomorphism_services, {"a": "x", "b": "z", "c": "y"})
    graph_isomorphism_services.core.checkers.revoke(
        checker_id, reason="force fail-closed integration case"
    )

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input=request_input,
        )
    )

    assert result.output["is_isomorphism"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["coverage"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None


def test_graph_isomorphism_accepts_graph_atlas_artifact_handoff(
    graph_isomorphism_services,
) -> None:
    searched = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.search.atlas",
            input={"order": 3, "constraints": {"connected": True}, "limit": 1},
        )
    )
    candidate = searched.output["candidates"][0]
    graph_uri = candidate["graph_uri"]
    vertices = candidate["graph"]["vertices"]

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": graph_uri,
                "right_graph_uri": graph_uri,
                "mapping": {vertex: vertex for vertex in vertices},
            },
        )
    )

    assert result.output["conclusion"] == "TRUE"
    assert result.output["left_graph_uri"] == graph_uri
    assert result.output["right_graph_uri"] == graph_uri
    assert graph_uri in result.artifact_uris
    pair = graph_isomorphism_services.core.store.get(result.output["graph_pair_uri"])
    assert pair.manifest.parents == (graph_uri,)


def test_graph_isomorphism_accepts_valid_unsorted_graph_artifacts(
    graph_isomorphism_services,
) -> None:
    left_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["c", "a", "b"],
        edges=[["b", "c"], ["a", "b"]],
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["z", "x", "y"],
        edges=[["y", "z"], ["x", "y"]],
    )

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": left_graph_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {"a": "x", "b": "y", "c": "z"},
            },
        )
    )

    assert result.output["conclusion"] == "TRUE"
    record = graph_isomorphism_services.core.store.get(
        result.output["verification_record_uri"]
    )
    assert left_graph_uri in record.manifest.parents
    assert right_graph_uri in record.manifest.parents


def test_graph_isomorphism_rejects_incompatible_graph_artifact(
    graph_isomorphism_services,
) -> None:
    wrong_artifact = graph_isomorphism_services.core.artifacts.put(
        schema_uri=graph_isomorphism_services.graph.scope_schema_uri,
        semantics_uri=graph_isomorphism_services.graph.semantics_uri,
        payload={
            "scope_schema_version": "1",
            "source": "networkx.graph_atlas_g",
            "backend_version": "test",
            "order": 3,
            "enumerated_count": 2,
        },
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["x"],
        edges=[],
    )

    result = graph_isomorphism_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": wrong_artifact.artifact_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {"x": "x"},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INCOMPATIBLE_GRAPH_ARTIFACT"
    assert result.diagnostics[0].path == "left_graph_uri"


def test_graph_isomorphism_is_unavailable_without_reference_checkers(
    unauthorized_graph_isomorphism_services,
) -> None:
    runtime = unauthorized_graph_isomorphism_services

    assert "graph.isomorphism.verify" not in {
        descriptor.operation_id
        for descriptor in runtime.core.operations.snapshot().operations
    }
