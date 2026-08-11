import networkx as nx

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.math import graphs


def _assert_computed_result(result) -> None:
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.artifact_uris == ()
    assert result.relationships == ()


def test_native_graph_metric_agrees_and_capability_remains_verifiable(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    graph = nx.path_graph(("a", "b", "c", "d"))
    computed = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.invariant.diameter.compute",
            input={
                "graph": {
                    "vertices": list(graph.nodes),
                    "edges": [list(edge) for edge in graph.edges],
                }
            },
        )
    )

    assert computed.output["result"]["diameter"] == graphs.diameter(graph)
    _assert_computed_result(computed)


def test_capability_provider_provenance_is_unchanged(
    authorized_complete_runtime,
) -> None:
    providers = {
        descriptor.capability_id: descriptor.provider
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id
        in {
            "graph.invariant.diameter.compute",
            "graph.invariant.diameter.verify",
        }
    }
    assert providers == {
        "graph.invariant.diameter.compute": "jacobian.graph-invariants",
        "graph.invariant.diameter.verify": "jacobian.graph-exact-checkers",
    }
