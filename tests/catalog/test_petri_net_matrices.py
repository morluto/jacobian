"""Catalog conformance for Petri-net matrix projection."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_matrices_example_returns_exact_pre_post_and_incidence() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("petri_net.matrices.compute")
    assert operation is not None
    example = operation.examples[0]

    result = invoke_operation(operation.operation_id, example.input, catalog)

    assert result.output["pre"]["entries"] == [["2"], ["0"]]
    assert result.output["post"]["entries"] == [["0"], ["3"]]
    assert result.output["incidence"]["entries"] == [["-2"], ["3"]]
    assert result.output["input_places_by_transition"] == [[0]]
    assert result.output["output_places_by_transition"] == [[1]]
    assert result.output["consumer_transitions_by_place"] == [[0], []]
    assert result.output["producer_transitions_by_place"] == [[], [0]]
    assert result.output["net"]["place_ids"] is None
    assert result.output["net"]["transition_ids"] is None
    assert result.output["net"]["pre"] == [[2], [0]]
    assert result.output["net"]["post"] == [[0], [3]]
