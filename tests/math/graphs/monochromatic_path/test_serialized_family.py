"""Colored hypergraph families retain source color and vertex axes."""

import json

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.monochromatic_path import (
    construct_monochromatic_path_hypergraphs,
    verify_monochromatic_path_hypergraphs,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def test_empty_edge_family_axes_and_claim() -> None:
    source = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=("a",), edges=()), edge_colors=()
    )
    result = construct_monochromatic_path_hypergraphs(source)
    assert result.colours == ("uncolored",)
    assert verify_monochromatic_path_hypergraphs(
        type(result).model_validate_json(result.model_dump_json())
    )
    payload = result.model_dump(mode="json")
    payload["colours"] = ["foreign"]
    with pytest.raises(ValidationError):
        type(result).model_validate_json(json.dumps(payload))
    payload = result.model_dump(mode="json")
    payload["hypergraphs"][0]["edges"] = []
    assert not verify_monochromatic_path_hypergraphs(
        type(result).model_validate_json(json.dumps(payload))
    )
    payload["hypergraphs"][0]["vertices"] = ["foreign"]
    with pytest.raises(ValidationError):
        type(result).model_validate_json(json.dumps(payload))
