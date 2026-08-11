from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.capabilities.graph_capabilities_support import (
    GraphTestServices,
    open_graph_services,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion


@pytest.fixture
def authorized_graph_services(tmp_path) -> Iterator[GraphTestServices]:
    with open_graph_services(tmp_path / "state", authorize_checker=True) as services:
        yield services


def test_degree_sequence_realization_materializes_replayable_graph(
    authorized_graph_services: GraphTestServices,
) -> None:
    result = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.realize.degree_sequence",
            input={"degree_sequence": [2, 2, 1, 1]},
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["conclusion"] == "GRAPHICAL"
    assert result.output["graph_uri"] in result.artifact_uris
    assert result.output["certificate_uri"] in result.artifact_uris
    verified = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.degree_sequence.verify",
            input={
                "certificate_uri": result.output["certificate_uri"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value


def test_degree_sequence_non_graphical_result_has_replayable_obstruction(
    authorized_graph_services: GraphTestServices,
) -> None:
    result = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.realize.degree_sequence",
            input={"degree_sequence": [3, 3, 1, 1]},
        )
    )

    assert result.output["conclusion"] == "NON_GRAPHICAL"
    assert result.output.get("graph_uri") is None
    assert result.output["obstruction"] == {
        "kind": "ERDOS_GALLAI",
        "k": 2,
        "lhs": 6,
        "rhs": 4,
    }
    verified = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.degree_sequence.verify",
            input={
                "certificate_uri": result.output["certificate_uri"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.FALSE.value
