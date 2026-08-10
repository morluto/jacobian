from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import Conclusion


def test_degree_sequence_realization_materializes_replayable_graph(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.realize.degree_sequence",
            input={"degree_sequence": [2, 2, 1, 1]},
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["conclusion"] == "GRAPHICAL"
    assert result.output["graph_uri"] in result.artifact_uris
    assert result.output["certificate_uri"] in result.artifact_uris
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            input={
                "certificate_uri": result.output["certificate_uri"],
                "checker_id": result.output["checker_id"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.TRUE.value


def test_degree_sequence_non_graphical_result_has_replayable_obstruction(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
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
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="certificate.verify",
            input={
                "certificate_uri": result.output["certificate_uri"],
                "checker_id": result.output["checker_id"],
            },
        )
    )
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.output["conclusion"] == Conclusion.FALSE.value
