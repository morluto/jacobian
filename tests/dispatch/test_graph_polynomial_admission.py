"""Graph polynomial native admission and dispatch preserve owner-local limits."""

from itertools import combinations
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.polynomials.operations import (
    chromatic_polynomial,
    flow_polynomial,
    tutte_polynomial,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


@pytest.mark.parametrize(
    ("kind", "native"),
    (
        ("chromatic", chromatic_polynomial),
        ("tutte", tutte_polynomial),
        ("flow", flow_polynomial),
    ),
)
@pytest.mark.parametrize("dimension", ("vertex_count", "edges"))
def test_graph_polynomial_native_dispatch_admission_agrees(
    kind: str, native: Any, dimension: str
) -> None:
    graph = IndexedSimpleUndirectedGraph(
        vertex_count=13 if dimension == "vertex_count" else 12,
        edges=()
        if dimension == "vertex_count"
        else tuple(combinations(range(8), 2))[:25],
    )
    with pytest.raises(OperationResourceAdmissionError) as native_error:
        native(graph)
    with pytest.raises(OperationResourceAdmissionError) as dispatch_error:
        invoke_operation(
            f"graph.polynomial.{kind}.compute",
            {"graph": graph.model_dump(mode="json")},
            Catalog.open(),
        )
    assert dispatch_error.value.errors() == native_error.value.errors()
    error = native_error.value.errors()[0]
    assert error["loc"] == ("graph", dimension)
    assert error["type"] == (
        "graph.polynomial.vertex_count_limit"
        if dimension == "vertex_count"
        else "graph.polynomial.edge_count_limit"
    )
    assert f"at most {12 if dimension == 'vertex_count' else 24}" in error["msg"]
    assert f"received {13 if dimension == 'vertex_count' else 25}" in error["msg"]
    assert "Submit a graph" in error["msg"]
