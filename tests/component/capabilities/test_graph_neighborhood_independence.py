from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.capabilities.graph_capabilities_support import (
    GraphTestServices,
    open_graph_services,
)

from jacobian.contracts.capabilities import (
    CapabilityInputKind,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion

_LEFT = tuple(range(6))
_RIGHT = tuple(range(6, 14))
_MISSING = {
    (0, 6),
    (0, 8),
    (1, 8),
    (1, 11),
    (2, 7),
    (2, 12),
    (3, 6),
    (3, 11),
    (4, 7),
    (4, 13),
    (5, 12),
    (5, 13),
}


@pytest.fixture
def authorized_graph_services(tmp_path) -> Iterator[GraphTestServices]:
    with open_graph_services(tmp_path / "state", authorize_checker=True) as services:
        yield services


def _wowii_200_graph() -> dict[str, object]:
    vertices = [str(vertex) for vertex in _LEFT + _RIGHT]
    edges = sorted(
        sorted((str(left), str(right)))
        for left in _LEFT
        for right in _RIGHT
        if (left, right) not in _MISSING
    )
    return {
        "graph_schema_version": "1",
        "vertices": sorted(vertices),
        "edges": edges,
    }


def test_neighborhood_independence_reproduces_wowii_200_invariant(
    authorized_graph_services: GraphTestServices,
) -> None:
    graph = authorized_graph_services.core.artifacts.put(
        schema_uri=authorized_graph_services.graph.graph_schema_uri,
        semantics_uri=authorized_graph_services.graph.semantics_uri,
        payload=_wowii_200_graph(),
        summary="WOWII Conjecture 200 public counterexample graph",
    )

    result = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.compute.neighborhood_independence",
            input={"graph_uri": graph.artifact_uri},
        )
    )

    assert result.output["total"] == 72
    assert result.output["average"] == {"num": "36", "den": "7"}
    assert len(result.output["records"]) == 14
    assert all(
        record["independence_number"] == len(record["neighborhood"])
        for record in result.output["records"]
    )
    assert result.output["certificate_uri"] in result.artifact_uris
    assert "conclusion" not in result.output

    verified = authorized_graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.neighborhood_independence.verify",
            input={
                "certificate_uri": result.output["certificate_uri"],
            },
        )
    )

    assert verified.verification_record_uri is not None
    assert verified.output["conclusion"] == Conclusion.TRUE.value
    assert verified.output["verification_record_uri"]

    descriptor = next(
        item
        for item in authorized_graph_services.core.capabilities.catalog().capabilities
        if item.capability_id == "graph.compute.neighborhood_independence"
    )
    assert descriptor.accepted_input_kinds == (CapabilityInputKind.TYPED_ARTIFACT,)
    assert descriptor.accepted_artifact_types == (
        authorized_graph_services.graph.graph_schema_uri,
    )
