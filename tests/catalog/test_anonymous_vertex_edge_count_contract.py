"""Discovery and schema contract for the anonymous vertex-deck quotient."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.decks.anonymous_vertex_edge_count import (
    AnonymousVertexDeckEdgeCount,
)

OPERATION_ID = "graph.deck.anonymous_vertex.edge_count.compute"


def test_result_field_is_named_for_the_presupposition() -> None:
    fields = AnonymousVertexDeckEdgeCount.model_fields
    assert "realizing_edge_count" in fields
    assert "implied_edge_count" not in fields
    description = fields["realizing_edge_count"].description
    assert description is not None
    assert "does not establish" in description
    assert "realizab" in description


def test_published_description_states_the_presupposition() -> None:
    tool = Catalog.open().operation(OPERATION_ID)
    assert tool is not None
    assert "realizing_edge_count" in tool.description
    assert "whether or not such a graph exists" in tool.description


def test_published_example_states_the_presupposition() -> None:
    tool = Catalog.open().operation(OPERATION_ID)
    assert tool is not None
    assert "does not itself establish realizability" in tool.examples[0].description


def test_operation_remains_published_and_discoverable() -> None:
    operation = Catalog.open().operation(OPERATION_ID)
    assert operation is not None
    assert operation.operation_id == OPERATION_ID
