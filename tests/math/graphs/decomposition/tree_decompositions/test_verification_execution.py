"""Decomposition verification cannot turn failed recomputation into disagreement."""

import pytest

from jacobian.math.graphs.decomposition.tree_decompositions import operations
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


@pytest.mark.parametrize(
    "operation",
    ["width", "vertex_occurrences", "adhesions", "reroot", "bag_intersection_graph"],
)
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError])
def test_decomposition_verifier_preserves_backend_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str, error_type: type[Exception]
) -> None:
    td = TreeDecomposition(
        graph=SimpleUndirectedGraph(vertices=("a",), edges=()),
        tree_nodes=("t",),
        tree_edges=(),
        bags=(("a",),),
    )
    args = (td, "t") if operation == "reroot" else (td,)
    claim = getattr(operations, operation)(*args)
    verify = getattr(operations, "verify_" + operation)
    assert verify(claim)

    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("backend failure")

    monkeypatch.setattr(operations, operation, fail)
    with pytest.raises(error_type, match="backend failure"):
        verify(claim)
