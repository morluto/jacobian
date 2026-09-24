"""Catalog examples for one-hole ranked-tree contexts."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_tree_context_examples_execute_through_catalog():
    catalog = Catalog.open()
    plug = catalog.operation("ranked_tree.context.plug.compute")
    state_map = catalog.operation("tree_automaton.context.state_map.compute")
    assert plug is not None and state_map is not None

    plugged = invoke_operation(plug.operation_id, plug.examples[0].input, catalog)
    mapped = invoke_operation(
        state_map.operation_id, state_map.examples[0].input, catalog
    )

    assert plugged.output["plugged_tree"] == {
        "symbol": 1,
        "children": [
            {"symbol": 0, "children": []},
            {"symbol": 0, "children": []},
        ],
    }
    assert mapped.output["state_map"] == [0]
