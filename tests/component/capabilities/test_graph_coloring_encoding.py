"""Graph-owned coloring encodings and independent replay tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import (
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs.coloring import install_graph_coloring_capabilities
from jacobian.runtime import CheckerAuthorityMode


@contextmanager
def _open_graph_coloring_services(
    root: Path,
    *,
    authorize_checker: bool,
) -> Iterator[DomainTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            adapters, _installation = install_graph_coloring_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.sat,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_capability(adapter)
        yield services


@pytest.fixture
def graph_coloring_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with _open_graph_coloring_services(
        tmp_path / "state", authorize_checker=True
    ) as services:
        yield services


@pytest.fixture
def unauthorized_graph_coloring_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    with _open_graph_coloring_services(
        tmp_path / "state", authorize_checker=False
    ) as services:
        yield services


def _encode(runtime: DomainTestServices) -> CapabilityResult:
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.coloring.encode_k_cnf",
            input={
                "graph": {
                    "vertices": ["c", "a", "b"],
                    "edges": [["b", "a"], ["c", "b"], ["a", "c"]],
                },
                "colors": 3,
            },
        )
    )


def test_graph_coloring_encoding_is_canonical_and_inspectable(
    unauthorized_graph_coloring_services,
) -> None:
    runtime = unauthorized_graph_coloring_services

    result = _encode(runtime)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["graph"] == {
        "graph_schema_version": "1",
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
    }
    assert result.output["variable_count"] == 9
    assert result.output["clause_count"] == 21
    assert result.output["checker_id"] is None
    assert len(result.artifact_uris) == 5


def test_graph_coloring_encoding_replays_through_domain_checker(
    graph_coloring_services,
) -> None:
    encoded = _encode(graph_coloring_services)

    assert encoded.output["checker_id"] is not None
    verified = graph_coloring_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.coloring.encoding.verify",
            input={
                "certificate_uri": encoded.output["certificate_uri"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["assurance"]["verification"] == "VERIFIED"
    assert verified.output["verification_record_uri"] is not None
