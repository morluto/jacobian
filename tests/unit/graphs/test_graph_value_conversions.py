from jacobian.contracts.graph_isomorphism import (
    SimpleUndirectedGraph as SimpleUndirectedGraphContract,
)
from jacobian.graphs.conversions import (
    graph_contract_from_value,
    graph_value_from_contract,
)
from jacobian.math.graphs import SimpleUndirectedGraph


def test_graph_value_and_wire_contract_round_trip_explicitly() -> None:
    value = SimpleUndirectedGraph(
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )

    contract = graph_contract_from_value(value)
    restored = graph_value_from_contract(contract)

    assert type(contract) is SimpleUndirectedGraphContract
    assert type(restored) is SimpleUndirectedGraph
    assert restored == value
    assert contract.model_dump(mode="json") == value.model_dump(mode="json")
