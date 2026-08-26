"""Transport admission at the tree-decomposition dispatch boundary."""

from __future__ import annotations

import pytest

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResult
from jacobian.dispatch import (
    OperationRequestValidationError,
    invoke_operation,
    parse_operation_input,
)
from jacobian.math.graphs.tree_decompositions._models import RerootRequest
from jacobian.math.graphs.tree_decompositions._operations import compute_reroot


def _path_decomposition(*, node_count: int, label_suffix: str) -> dict[str, object]:
    nodes = [f"n{index:03d}_{label_suffix}" for index in range(node_count)]
    return {
        "graph": {"vertices": ["a"], "edges": []},
        "tree_nodes": nodes,
        "tree_edges": [
            [nodes[index], nodes[index + 1]] for index in range(node_count - 1)
        ],
        "bags": [["a"] for _ in nodes],
    }


def test_reroot_transport_admission_round_trips_canonical_projection() -> None:
    """A bounded source survives serialize, parse, projection, and dispatch."""

    decomposition = _path_decomposition(node_count=6, label_suffix="label")
    tree_nodes = decomposition["tree_nodes"]
    assert isinstance(tree_nodes, list)
    payload = {"decomposition": decomposition, "root": tree_nodes[0]}

    serialized = encode_strict_json(payload)
    request = parse_operation_input(RerootRequest, payload)
    assert request == RerootRequest.model_validate_json(serialized)

    projected = compute_reroot(request).model_dump(mode="json")
    projected_bytes = len(encode_strict_json(projected))
    assert projected_bytes <= CanonicalLimits().max_output_bytes

    public_result = invoke_operation(
        "graph.tree_decomposition.reroot.compute", payload, Catalog.open()
    )
    assert (
        OperationResult.model_validate_json(public_result.model_dump_json())
        == public_result
    )
    assert public_result.output == projected


def test_reroot_rejects_paths_that_exceed_transport_before_execution() -> None:
    """A small request whose repeated path labels overflow is not admitted."""

    decomposition = _path_decomposition(node_count=256, label_suffix="x" * 394)
    tree_nodes = decomposition["tree_nodes"]
    assert isinstance(tree_nodes, list)
    payload = {"decomposition": decomposition, "root": tree_nodes[0]}
    assert len(encode_strict_json(payload)) <= CanonicalLimits().max_input_bytes

    with pytest.raises(OperationRequestValidationError) as exc_info:
        invoke_operation(
            "graph.tree_decomposition.reroot.compute", payload, Catalog.open()
        )

    assert exc_info.value.errors()[0]["type"] == (
        "graph.reroot_result_exceeds_transport_limit"
    )
